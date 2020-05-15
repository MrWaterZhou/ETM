#!/bin/bash
location=$2
function getdir() {
    lang="zh-CN"
    for element in $(ls $1); do
        dir_or_file=$1"/"$element
        if [ -d $dir_or_file ]; then
            getdir $dir_or_file
        else
	    # echo $dir_or_file

            if [[ $dir_or_file == *$lang* ]]; then
                #cat $dir_or_file | egrep '"source":"chat"' | sed -r 's/\{"aTag":"(.*)","aid":(.*),"from":"(.*)","id":(.*),"kid":(.*),"lang":"(.*)","lang_user":"(.*)","msg":"(.*)","mtime":(.*),"pid":(.*),"source":"(.*)","to":"(.*)","type":"(.*)","userName":"(.*)","vipLevel":(.*)\}/\3 \8/g'
                cat $dir_or_file >> $location
            fi
        fi
    done
}

getdir $1