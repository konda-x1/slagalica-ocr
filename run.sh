#!/usr/bin/env bash

count() { echo $#; }

via_urls ()
{
    youtube-dl -f "bestvideo[height<=720]/best[height<=720]" -o "downloads/%(id)s" \
        --write-info-json --ignore-errors "$@"
    filenames=`python3 url_video_ids.py "downloads/" "$@"`
    localvids=''
    if ! [ -z "$filenames" ]
    then
        newfilenames=`python3 info_rename.py $filenames`
        if ! [ -z "$newfilenames" ]
        then
            mkdir -p local
            localvids=`mv -v $newfilenames local/ | sed 's/.*-> '\''\(.*\)'\''/\1/'`
            mvjsons=''; for i in $newfilenames; do mvjsons="$mvjsons $i.info.json"; done
            mv $mvjsons local/
        fi
    fi
    via_filenames $localvids
}

via_filenames ()
{
    python3 extract.py "$@"
}

cmd="$1"

if [ "$cmd" == "sync" ]
then
    echo "Downloading playlist info..."
    youtube-dl -j --flat-playlist --playlist-reverse "`cat playlist_url.txt`" \
        | jq -r '"https://www.youtube.com/watch?v=\(.id)"' > playlist.txt
    echo "Done"
elif [ "$cmd" == "local" ]
then
    if [ $# -lt 2 ]
    then
        >&2 echo "error: no file names supplied"
        exit -2
    fi
    via_filenames "${@:2}"
elif [ "$cmd" == "urls" ]
then
    if [ $# -lt 2 ]
    then
        >&2 echo "error: no URLs supplied"
        exit -2
    fi
    via_urls "${@:2}"
elif [ "$cmd" == "random" ]
then
    urls=`python3 randlines.py playlist.txt $2`
    ecode=$?
    if [ "$ecode" -eq "0" ]
    then
        via_urls $urls
    elif [ "$ecode" -eq "253" ]
    then
        >&2 echo "Did you forget to run the 'sync' command?"
        exit -1
    fi
elif [ "$cmd" == "all" ]
then
    [ -s "playlist.txt" ] && { urls=`cat playlist.txt` && ! [ -z "$urls" ]; } \
    && via_urls $urls \
    || { >&2 echo -e "error: 'playlist.txt' is nonexistant or empty.\nDid you forget to run the 'sync' command?"; exit -1; }
else
    >&2 echo -e "error: Unknown command '$cmd'\nSupported commands: sync, local, urls, random, all"
    exit -1
fi
