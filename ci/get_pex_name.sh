function get_pex_name(){
    if [[ -z ${CIRCLECI+x} ]]; then
        BRANCH=$(git rev-parse --abbrev-ref HEAD)
        SHA=$(git rev-parse --short HEAD)
        EXTRA="LOCAL_${BRANCH}"
    else
        NUM=$(printf %06.0f $CIRCLE_BUILD_NUM)
        SHA="${CIRCLE_SHA1:0:7}"
        if [[ -z ${CIRCLE_PR_NUMBER+x} ]]; then
            EXTRA="${NUM}_MASTER"
        else
            EXTRA="${NUM}_PR_${CIRCLE_PR_NUMBER}"
        fi
    fi

    echo "skynet_${EXTRA}_$(date +%Y-%m-%d_%Hh%M)_${SHA}.pex"
}

"$@"
