#! /bin/sh

make interactive -j

bin/interactive -e 1 -i data/ -d 20,200,50 -b 4,20,10 -p 0.01 -m 200.271 \
	-x 927.426 "$@" -o results/ -f statistics.csv

bin/interactive -e 3 -i /zfs
