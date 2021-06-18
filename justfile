# vim: set ft=make :

build:
	#!/usr/bin/env sh
	rm -r post-formatted
	cp -r post post-formatted
	mmv post-formatted/\* post-formatted/\#1.html



