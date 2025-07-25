#!/bin/bash

# 设置 Playwright 环境变量
export TMPDIR=/rt-vepfs/wqs/workspace/RAGEN/tmp
export PLAYWRIGHT_BROWSERS_PATH=/rt-vepfs/wqs/workspace/RAGEN/playwright_browsers

# 创建必要的目录
mkdir -p $TMPDIR
mkdir -p $PLAYWRIGHT_BROWSERS_PATH

echo "✅ Playwright 环境变量已设置:"
echo "   TMPDIR=$TMPDIR"
echo "   PLAYWRIGHT_BROWSERS_PATH=$PLAYWRIGHT_BROWSERS_PATH" 