#!/bin/bash
# StockBEE — 开发环境初始化
# Clone 后跑一次: ./setup.sh

set -e

echo "Setting up StockBEE dev environment..."

# ---------------------------------------------------------------
# 1. 链接共享 skills 到 .claude/skills
#    (Claude Code 从 .claude/skills 加载项目级 skills;
#     .claude/ 在 .gitignore 里,所以 symlink 每次 clone 后重建)
# ---------------------------------------------------------------
mkdir -p .claude
if [ -L .claude/skills ]; then
  echo "✓ .claude/skills symlink already exists"
elif [ -d .claude/skills ]; then
  echo "⚠ .claude/skills is a real directory, replacing with symlink..."
  rm -rf .claude/skills
  ln -s ../claude-skills .claude/skills
  echo "✓ .claude/skills → claude-skills/"
else
  ln -s ../claude-skills .claude/skills
  echo "✓ .claude/skills → claude-skills/"
fi

# ---------------------------------------------------------------
# 2. Python venv (.venv/) + 安装依赖
# ---------------------------------------------------------------
if [ ! -d .venv ]; then
  python3 -m venv .venv
  echo "✓ Created .venv/"
else
  echo "✓ .venv/ already exists"
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip >/dev/null
echo "✓ pip upgraded"

if [ -f requirements.txt ]; then
  pip install -r requirements.txt
  echo "✓ requirements.txt installed"
else
  echo "⚠ requirements.txt not found; install project deps manually as needed"
fi

# ---------------------------------------------------------------
# 3. 冒烟测试: 可以 import 项目包
# ---------------------------------------------------------------
if PYTHONPATH=src python -c "import stockbee" 2>/dev/null; then
  echo "✓ stockbee package importable"
else
  echo "⚠ stockbee package import failed; check src/stockbee/ + deps"
fi

echo ""
echo "Done! Next steps:"
echo "  1. Activate venv:   source .venv/bin/activate"
echo "  2. Run tests:       PYTHONPATH=src pytest tests/ -q"
echo "  3. Claude skills:   ls .claude/skills   (→ ../claude-skills)"
