#!/bin/sh

cd ..
cd ..
sh ./sync_with_desktop.sh
cd GPUPIC/operatorcosts

ssh josh-desktop 'cd /home/josh/GPUPIC/GPUPIC/operatorcosts; gmake; ldd a.out'
