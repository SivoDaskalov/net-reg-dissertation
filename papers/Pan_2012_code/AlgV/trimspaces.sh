#!/bin/bash

sed -i '/^sim=/d' setup*.txt
sed -i 's/^ //g' setup*.txt
sed -i 's/ \+/,/g' setup*.txt
