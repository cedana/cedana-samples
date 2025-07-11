#!/bin/sh

trap 'echo "Interrupted"; exit 1' INT TERM

while :; do
    sleep 1
    date
done
