import datetime
import functools
import glob
import json
import os

import requests
import yaml
from fabric.api import abort, env, hide, put, run, settings, sudo
from fabric.contrib.console import confirm
from fabric.contrib.files import exists
from fabric.operations import prompt
from tabulate import tabulate

from scripts.__main__ import SERVICES

SERVICES_SCRIPTS = " ".join(SERVICES)

env.forward_agent = True
env.use_ssh_config = True

# allow specifying sudo user via command line -u flag
SUDO_USER = env.user
env.user = "deploy"

ORG = "AmbiLabs"
REPO = "ambi_brain"
CONFIG = "config.yml"

LOCAL_DAGS_DIR = os.path.join("airflow/dags")
DAG_NAMES = [
    os.path.splitext(os.path.basename(p))[0]
    for p in glob.glob(os.path.join(LOCAL_DAGS_DIR, "*.py"))
]
RUN_TIMESTAMP = datetime.datetime.utcnow()

# XXX edit this to ambiclimate aftermigration
DOMAINS = {"staging": "sg.tryambi.com", "production": "sg.ambi-labs.com"}
HOSTS_STAGING = {"train": ["train1.ai"], "pred": ["pred1.ai"]}
HOSTS_PRODUCTION = {
    "train": ["train1.ai"],
    "pred": [f"pred{i}.ai" for i in range(1, 4)],
}
HOSTS = {"staging": HOSTS_STAGING, "production": HOSTS_PRODUCTION}


env.base_dir = "/opt/ai/"
env.pex_dir = os.path.join(env.base_dir, "shared", "pex")
env.deployed = []


def staging():
    env.stage = "staging"
    _stage_setup()


def production():
    env.stage = "production"
    _stage_setup()


def _stage_setup():
    domain = DOMAINS[env.stage]
    env.hosts = env.hosts or [
        f"{host}.{domain}" for hosts in HOSTS[env.stage].values() for host in hosts
    ]
    with open("config.yml") as f:
        env.slack = yaml.safe_load(f)["slack"][env.stage]


def train():
    env.hosts = [h for h in env.hosts if h.startswith("train")]
    env.pex_owner = "airflow"
    env.pex_extract_dir = "/var/lib/airflow/.pex/install"


def pred():
    env.hosts = [h for h in env.hosts if h.startswith("pred")]
    env.pex_owner = "deploy"
    env.pex_extract_dir = "/home/deploy/.pex/install"


def _common_config():
    env.shared_config_path = os.path.join(env.app_dir, "config.yml")
    env.current_release_link = os.path.join(env.app_dir, "current")


def service_config(service_name):
    env.service_name = service_name
    env.unit = service_name
    env.app_dir = os.path.join(env.base_dir, "services", service_name)
    env.releases_dir = os.path.join(env.app_dir, "releases")
    _common_config()


def dag_config(dag_name):
    env.dag_name = dag_name
    env.unit = dag_name
    airflow_dir = os.path.join(env.base_dir, "airflow")
    env.app_dir = os.path.join(airflow_dir, "dags", dag_name)
    env.releases_dir = os.path.join(airflow_dir, "releases", dag_name)
    _common_config()


def _force_link(target, link_name):
    run(f"rm {link_name} || true")
    run(f"ln -sf {target} {link_name}")


def _register_action(action, host, unit, release):
    env.deployed.append(
        {
            "action": action,
            "host": host,
            "unit": unit,
            "release": os.path.basename(release),
        }
    )


def create_release():
    get_release_tag()
    # create new release directory
    run(f"mkdir -p {env.latest_release_dir}")

    # link skynet.pex in new release dir to the PEX file
    env.pex_link_path = os.path.join(env.latest_release_dir, "skynet.pex")
    env.pex_path = os.path.join(env.pex_dir, env.latest_pex)
    run(f"ln -sf {env.pex_path} {env.pex_link_path}")

    # copy the configuration file (-p to preserve permissions)
    run(f"cp -p {env.shared_config_path} {env.latest_release_dir}")

    # link current release to the release we just created
    _force_link(env.latest_release_dir, env.current_release_link)
    _register_action("release", env.host, env.unit, env.latest_release_dir)


def _pred_services():
    return SERVICES


def deploy(service=None, build_num=None, dag_name=None):

    install_pex(build_num=build_num)

    if env.host_string.startswith("train"):
        airflow_dag_release(dag_name)
    if env.host_string.startswith("pred"):
        if service is None:
            services = _pred_services()
        else:
            services = [service]
        for service_name in services:
            service_config(service_name)
            create_release()

    notify_slack()


def release(service_name=None):
    if service_name is None:
        services = _pred_services()
    else:
        services = [service_name]
    for service in services:
        service_config(service)
        create_release()


def _get_release_tag(pex_path):
    pex_tag = os.path.splitext(os.path.basename(env.latest_pex))[0]
    date_tag = RUN_TIMESTAMP.strftime("%Y-%m-%d_%Hh%Mm%S")
    return f"{pex_tag}_at_{date_tag}"


def get_pex_files():
    """Initialize configuration."""
    env.pex_files = run(f"ls -t {env.pex_dir}").split()


def get_release_tag():
    get_pex_files()
    if not env.pex_files:
        raise ValueError(f"no releases found in {env.pex_dir}")

    env.latest_pex = env.pex_files[0]
    release_tag = _get_release_tag(env.latest_pex)
    env.latest_release_dir = os.path.join(env.releases_dir, release_tag)


def get_releases():
    with hide("stdout"):
        env.releases = sorted(run(f"ls -x {env.releases_dir}").split(), reverse=True)


def rollback(service=None, dag=None, n_show=10):

    if service is not None:
        config = service_config
        if service == "all":
            units = _pred_services()
        else:
            units = [service]
    elif dag is not None:
        config = dag_config
        if dag == "all":
            units = DAG_NAMES
        else:
            units = [dag]
    else:
        raise ValueError("provide service or dag ('all' for all)")

    for unit in units:
        config(unit)
        get_releases()
        print(
            f"Releases for {unit} (total {len(env.releases)}, "
            "increase n_show to see more)\n"
        )
        for idx, release in enumerate(env.releases[: int(n_show)]):
            print(f"{idx:4}", release)
        print()
        idx = int(prompt("Rollback to? Ctrl + C to abort."))
        rollback_path = os.path.join(env.releases_dir, env.releases[idx])
        _force_link(rollback_path, env.current_release_link)
        _register_action("rollback", env.host, env.unit, rollback_path)
        print(f"rolled back to {rollback_path}")

    notify_slack()


def remove_extracted_pex_artifacts():
    cmd = f"rm -r {env.pex_extract_dir}"
    if env.pex_owner == env.user:
        run(cmd)
    else:
        with settings(user=SUDO_USER):
            sudo(cmd, user=env.pex_owner)


def clean(n_keep=10):
    """Clean all but n_keep=10 releases."""
    n_keep = int(n_keep)
    get_pex_files()
    print("keep:")
    to_keep = env.pex_files[:n_keep]
    to_delete = env.pex_files[n_keep:]
    if to_delete:
        for r in to_keep:
            print("\t" + r)
        print("delete:")
        for r in to_delete:
            print("\t" + r)
        if confirm("delete?", default=False):
            if len(to_delete) > 1:
                basenames = "{" + ",".join(to_delete) + "}"
            else:
                basenames = to_delete[0]
            run(f"rm {env.pex_dir}/{basenames}")
    else:
        abort(f"Only {len(to_keep)} releases installed.")


def extract_airflow_dag(dag_name):
    """Extract dag from pex file into release directory"""
    dag_pattern = os.path.join("*", LOCAL_DAGS_DIR, f"{dag_name}.py")
    run(
        f"tail -n +2 {env.pex_path} | bsdtar -xv -C {env.latest_release_dir} "
        f"--strip-components 4 -f - {dag_pattern}"
    )


def airflow_dag_release(dag_name=None):
    if dag_name is None:
        dags = DAG_NAMES
    else:
        dags = [dag_name]
    for dag in dags:
        print(f"making DAG release for {dag}")
        dag_config(dag)
        create_release()
        extract_airflow_dag(dag)


def start(script):
    """Start one service."""
    run(f"supervisorctl start {script}")


def restart(script):
    """Restart one service."""
    run(f"supervisorctl restart {script}")


def stop(script):
    """Stop one service."""
    run(f"supervisorctl stop {script}")


def restart_all():
    run(f"supervisorctl restart {SERVICES_SCRIPTS}")


def start_all():
    run(f"supervisorctl start {SERVICES_SCRIPTS}")


def stop_all():
    run(f"supervisorctl stop {SERVICES_SCRIPTS}")


def status():
    run("supervisorctl status")


def deploy_local_pex(fname):
    basename = os.path.basename(fname)
    pex_path = os.path.join(env.pex_dir, basename)
    if confirm(f"Deploy\n\t{fname}\nto\n\t{pex_path} ?", default=False):
        put(fname, pex_path, mode=0o700)
        print(f"PEX installed at {pex_path}")


def install_pex(build_num=None, yes=False):
    """Download pex artifact to server.

    Parameters
    ----------
    build_num: int (default=None)
        download artifact for build, default: last master build

    Notes
    -----
    CirleCI token needs to be put into circle_client.txt.
    Requires circleclient: pip install circleclient
    """
    try:
        from circleclient.circleclient import CircleClient
    except ImportError:
        abort(
            "\nCouldn't import circleclient. To install, run: "
            "\n\n\tpip install circleclient"
        )

    try:
        with open("circle_token.txt") as f:
            token = f.read().strip()
    except IOError:
        abort("Put circleci token into circle_token.txt")

    client = CircleClient(api_token=token)

    if build_num is None:
        build = client.build.recent(ORG, REPO, limit=1, branch="master")[0]
        build_num = build["build_num"]
    else:
        build = client.build.status(ORG, REPO, build_num)

    artifacts = client.build.artifacts(ORG, REPO, build_num)
    pex_artifacts = [a for a in artifacts if a["path"].endswith(".pex")]

    if not pex_artifacts:
        abort("no pex file in artifacts")
    elif len(pex_artifacts) != 1:
        print([a["url"] for a in pex_artifacts])
        abort("multiple pex files found")
    else:
        pex = pex_artifacts[0]

    basename = os.path.basename(pex["path"])
    pex_path = os.path.join(env.pex_dir, basename)

    if exists(pex_path):
        print(f"PEX {pex_path} is already installed.")
        return

    if yes or confirm(
        f"deploy build {build_num}: {build['subject']}\n" f"PEX {basename}",
        default=False,
    ):

        put("circle_token.txt", "~/circle_token.txt", mode=0o600)
        cmd = (
            'python -c "import urllib;'
            "filehandle = urllib.FancyURLopener();"
            "token = open('circle_token.txt').read().strip();"
            "filehandle.retrieve('{}?circle-token={{}}'.format(token),'{}');\""
            "".format(pex["url"], pex_path)
        )
        run(cmd)
        run(f"chmod 755 {pex_path}")
    else:
        abort("Deployment cancelled by user.")


def runs_final(func):
    @functools.wraps(func)
    def decorated(*args, **kwargs):
        if env.host_string == env.all_hosts[-1]:
            return func(*args, **kwargs)

        else:
            return None

    return decorated


@runs_final
def notify_slack():
    url = env.slack["webhook_url"]
    headers = {"content-type": "application/json"}
    without_dupes = [env.deployed[0]] + [
        {k: v for k, v in b.items() if a[k] != v}
        for a, b in zip(env.deployed[:-1], env.deployed[1:])
    ]
    message = tabulate(without_dupes, headers="keys")
    print(message)
    payload = dict(attachments=[dict(color="good", title="Deployment", text=message)])
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    print(response.text)
