#!/usr/bin/env sh

# Set up BindCraft env
git clone https://github.com/martinpacesa/BindCraft
bash BindCraft/install_bindcraft.sh --cuda '12.4' --pkg_manager 'conda'
