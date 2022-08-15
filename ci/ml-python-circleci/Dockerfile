#
#   AMBI BRAIN Base Python Image
#
FROM ubuntu:16.04
LABEL ubuntu=16.04 python=3.6.7

# Add CI arg for mirror replacement
ARG CI

# Python 3 stuff tends to fall over without proper locale
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

ADD build.sh /tmp

RUN bash /tmp/build.sh
USER circleci
