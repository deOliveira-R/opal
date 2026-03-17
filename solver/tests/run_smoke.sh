#!/bin/zsh
/Users/rodrigo/git/OPAL/external/venv/bin/python /Users/rodrigo/git/OPAL/solver/tests/smoke_test.py > /tmp/opal_smoke.out 2>&1
echo "exit=$?" >> /tmp/opal_smoke.out
