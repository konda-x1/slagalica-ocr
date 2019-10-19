import os
import sys
import json
import re
import random
import shutil

def get_info_dict(filename):
    with open(filename) as jsonfile:
        info = json.load(jsonfile)
    return info

def get_season_episode(info, filename):
    results = re.findall(r'\d+', info['description'])[:2]
    season = results[0] if len(results) >= 1 else "000"
    episode = results[1] if len(results) >= 2 else "00"
    got_episode = len(results) >= 2
    if season == "000" or episode == "00":
        episode = "%02d" % random.randrange(0, 100)
        print("Could not detect both season and episode number from video description of video '%s'" % filename)
    return season, episode

def get_date(info, filename):
    title = info["fulltitle"]
    dateparts = re.findall(r'\d+', title)[:3]
    date = ''
    if len(dateparts) < 3:
        print("Could not detect full airing date from video title of video '%s'. Using upload date instead." % filename)
        udate = info["upload_date"]
        date = "%s. %s. %s." % (udate[6:], udate[4:6], udate[:4])
    else:
        date = '%02d. %02d. %04d.' % (int(dateparts[0]), int(dateparts[1]), int(dateparts[2]))
    return date

def create_empty_file(filename):
    open(filename, 'a').close()

def get_video_dict(info, season, episode, filename):
    d = dict()
    d["season"] = season
    d["episode"] = episode
    d["date"] = get_date(info, filename)
    d["url"] = info["webpage_url"]
    return d

filenames = sys.argv[1:]
newnames = []
for fn in filenames:
    filename = fn + ".info.json"
    info = get_info_dict(filename)

    season, episode = get_season_episode(info, filename)
    newname = "c%03de%02da" % (int(season), int(episode))

    # If episode was already processed before
    processed = os.path.isfile(r"out/" + newname + ".txt")
    if not processed:
        dir = re.findall(r'.*/|$', filename)[0]
        fullnewname = dir + newname
        newnames.append(fullnewname)

        #os.rename(fn, fullnewname)
        #create_empty_file(fn)
        shutil.copyfile(fn, fullnewname)

        d = get_video_dict(info, season, episode, filename)
        with open(fullnewname + ".info.json", "w+") as jsonfile:
            json.dump(d, jsonfile)
        #newnames.append(fullnewname + ".info.json")

print(*newnames)
sys.exit(0)
