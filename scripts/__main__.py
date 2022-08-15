import os

# set locale if unset for click
# see http://click.pocoo.org/5/python3/#python-3-surrogate-handling for details
# Set environment variable to utf8 by default, to fix logging errors
os.environ["LANG"] = "C.UTF-8"
os.environ["LC_ALL"] = "C.UTF-8"

import click  # noqa, pylint: disable=wrong-import-position


SERVICES = [
    "analytics_service",
    "comfort_service",
    "control",
    "control_worker",
    "status_monitor",
    "control_service_failover",
    *[
        "prediction_service_{}".format(model)
        for model in ["climate", "mode", "comfort"]
    ],
    "disaggregation_service",
]


class CLI(click.MultiCommand):
    def list_commands(self, ctx):
        return sorted(SERVICES)

    def get_command(self, ctx, name):
        ns = {}
        fn = os.path.join(os.path.dirname(__file__), name + ".py")
        with open(fn) as f:
            code = compile(f.read(), fn, "exec")
            eval(code, ns, ns)  # pylint: disable=eval-used
        return ns["main"]


main = CLI(help="Skynet entry point")

if __name__ == "__main__":
    main()
