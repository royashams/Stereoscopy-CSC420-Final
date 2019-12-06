#!/bin/sh
echo "sup"
cd ./MiddEval3/alg-ELAS
./run veronica_left.png veronica_right.png 2 results
cp -p ./results/disp0.pfm ./../../results