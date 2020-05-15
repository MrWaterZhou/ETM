log_path=$1
save_path=$2

#rm -rf $save_path
#mkdir $save_path
# get log
#sh extract_log.sh $log_path $save_path/all.log
cat $save_path/all.log | parallel --pipe -k  jq .id > $save_path/id.tmp
cat $save_path/all.log | parallel --pipe -k  jq .msg > $save_path/msg.tmp
cat $save_path/all.log | parallel --pipe -k  jq .from > $save_path/from.tmp
paste $save_path/id.tmp $save_path/from.tmp $save_path/msg.tmp > $save_path/msg.log
rm $save_path/id.tmp $save_path/from.tmp $save_path/msg.tmp

# sort and uniq
sort -n $save_path/msg.log | uniq > $save_path/msg.log.sorted
cat $save_path/msg.log.sorted | cut -f 3 | sed -r 's/"//g' > $save_path/msg.txt
sed -i -r 's/#[0-9]{3}//g' $save_path/msg.txt

# segment
cat $save_path/msg.txt | parallel --pipe -k ./THULAC/thulac -model_dir ./THULAC/models -t2s > $save_path/msg.segment
# prepare data
## get vocab
#cat $save_path/msg.segment | sed -r 's/\_[a-z]+//g' | sed 's/ /\n/g' | sed '/^$/d' | sort | uniq -c | sort > $save_path/vocab.freq





