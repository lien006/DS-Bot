#!/bin/bash
# 开发版 DeepSeek Bot 启动脚本：使用隔离数据目录，不影响正式版
# 用法：bash /Users/polyv/Documents/DS-Bot/start-dev.sh
unset ELECTRON_RUN_AS_NODE
export DS_BOT_HOME=/Users/polyv/Documents/dsbot-dev-data
export DS_BOT_WORKSPACE=/Users/polyv/Documents/dsbot-dev-workspace
cd "$(dirname "$0")/app"
exec node_modules/electron/dist/Electron.app/Contents/MacOS/Electron .
