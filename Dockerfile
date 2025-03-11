# This image is used in CI to run cedana-sample tests.
# This container itself is checkpoint/restored, which gives us some flexibility
# when deciding what to run.
#
# TODO:
#   - PVC for weights?

FROM ubuntu:22.04
LABEL maintainer="Niranjan Ravichandra <nravic@cedana.ai>"

# Grab cedana-samples workdir
WORKDIR /app
COPY . .

# Install dependencies
