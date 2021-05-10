# vim: set ft=make :

build:
	#!/usr/bin/env sh
	cp -r post post-formatted
	mmv post-formatted/\* post-formatted/\#1.html



