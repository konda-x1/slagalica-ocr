import sys
import numpy as np
import cv2
import random
import os
import errno
import pytesseract
import json

def mkdir(value):
    if not os.path.exists(value):
        try:
            os.mkdir(value)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def get_info_dict(filename):
    with open(filename) as jsonfile:
        info = json.load(jsonfile)
    return info

def showframes(*frames):
    for i, frame in enumerate(frames):
        cv2.imshow("image" + str(i+1), frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def imagedist(ima, imb, mask, nonzero_pixelcount):
    assert (ima * mask == ima).all()
    #assert (imb * mask == imb).all()
    imc = (ima - imb) ** 2
    #roots = np.sum(imc, -1) ** 0.5
    #dist = np.sum(roots) / nonzero_pixelcount
    dist = np.sum(imc) / nonzero_pixelcount
    return dist

def process_image(img):
    greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = cv2.resize(greyscale, (0, 0), None, 5, 5, cv2.INTER_LANCZOS4)
    res, threshold = cv2.threshold(scale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5,5), np.uint8)
    erosion = cv2.erode(threshold, kernel, iterations = 1)
    return erosion

def im2str(img):
    return pytesseract.image_to_string(img, config="--psm 7", lang="srp")

def get_autocrop_coordinates(img, background = None):
    if type(background) == type(None):
        background = img[0][0]
    w = img.shape[1]
    h = img.shape[0]
    min_x, min_y = w-1, h-1
    max_x, max_y = 0, 0
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            if any(img[y][x] != background):
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y
    return min_y, max_y, min_x, max_x

def crop(img, min_y, max_y, min_x, max_x):
    return img[min_y : max_y + 1, min_x : max_x + 1].copy()



class EpisodeMasks(object):
    mask_types = [1, 2]

    def __init__(self, mask_type):
        if mask_type not in self.mask_types:
            raise ValueError("invalid mask type")
        self.mask_type = mask_type
        self._load_masks()
        self._configure()

    def _set_orig_pixelcount(self):
        self.orig_pixelcount = np.count_nonzero(self.detect) // 3

    def _load_masks(self):
        def readmask(masknum, maskkey, ret_orig = False):
            #print(r"masks/mask" + str(masknum) + str(maskkey) + ".png")
            orig = cv2.imread(r"masks/mask" + str(masknum) + str(maskkey) + ".png")
            #showframes(orig)

            mask = orig.copy() if ret_orig else orig
            mask[np.any(mask != [0, 0, 0], axis=-1)] = [1, 1, 1]
            return (orig, mask) if ret_orig else mask

        self.orig, self.detect = readmask(self.mask_type, "", True)
        self.fields = dict()
        for a in ["A", "B", "V", "G"]:
            for b in ["1", "2", "3", "4", ""]:
                key = a + b
                self.fields[key] = readmask(self.mask_type, key)
        self.fields["K"] = readmask(self.mask_type, "K")

    def _configure(self):
        self._set_orig_pixelcount()
        crops_filename = r"masks/mask" + str(self.mask_type) + "crops.npy"
        try:
            self.field_crops = np.load(crops_filename, allow_pickle=True).item()
        except FileNotFoundError:
            self.field_crops = dict()
            for key, msk in self.fields.items():
                self.field_crops[key] = get_autocrop_coordinates(msk, [0, 0, 0])
            try:
                np.save(crops_filename, self.field_crops, allow_pickle=True)
            except OSError:
                print("Failed to save field crops for mask_type = %d." % self.mask_type)

    @property
    def framewidth(self):
        return self.orig.shape[1]

    @property
    def frameheight(self):
        return self.orig.shape[0]

    def cropped_field(self, img, field_key):
        if field_key not in self.fields:
            raise ValueError("invalid field key")
        min_y, max_y, min_x, max_x = self.field_crops[field_key]
        return crop(img, min_y, max_y, min_x, max_x)

    def resize(self, framewidth, frameheight):
        Rx = framewidth / self.framewidth
        Ry = frameheight / self.frameheight

        # Resize masks
        self.orig = cv2.resize(self.orig, (framewidth, frameheight))
        self.detect = cv2.resize(self.detect, (framewidth, frameheight), interpolation=cv2.INTER_NEAREST)
        for key, msk in self.fields.items():
            self.fields[key] = cv2.resize(msk, (framewidth, frameheight), interpolation=cv2.INTER_NEAREST)

        # Scale crops
        for key, val in self.field_crops.items():
            self.field_crops[key] = (round(Ry * val[0]), round(Ry * val[1]), round(Rx * val[2]), round(Rx * val[3]))

    def resized_copy(self, framewidth, frameheight):
        copy = EpisodeMasks(self.mask_type)
        if framewidth != self.framewidth or frameheight != self.frameheight:
            copy.resize(framewidth, frameheight)
        return copy

class EpisodeProcessor(object):
    def __init__(self):
        self._masks = dict()
        for t in EpisodeMasks.mask_types:
            self._masks[t] = None

    def _loaded_masks(self, mask_type):
        assert mask_type in EpisodeMasks.mask_types
        return type(self._masks[mask_type]) != type(None)

    def _format_episode_params(self, season, episode, date, url):
        season = "%03d" % int(season)
        episode = "%02d" % int(episode)
        date = str(date)
        url = str(url)
        return season, episode, date, url

    def _get_video_params(self, cap):
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        framerate = cap.get(cv2.CAP_PROP_FPS)
        return width, height, total_frames, framerate

    def _load_masks(self, mask_type):
        assert mask_type in EpisodeMasks.mask_types
        self._masks[mask_type] = EpisodeMasks(mask_type)

    def get_masktype(self, season, episode):
        s_num = int(season)
        e_num = int(episode)
        if s_num < 109:
            return 1
        else:
            return 2

    def _get_mask(self, season, episode, framewidth, frameheight):
        mask_type = self.get_masktype(season, episode)
        if not self._loaded_masks(mask_type):
            self._load_masks(mask_type)
        return self._masks[mask_type].resized_copy(framewidth, frameheight)

    def _process_frames(self, cap, total_frames, framerate, mask, delta_secs = 3.0):
        fi = total_frames / 2
        i = int(fi)
        start_frame = int(fi)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        cap.grab()

        prev_i = i
        frames = []
        dists = []
        while(i < total_frames):
            assert i - prev_i >= 0
            for j in range(i - prev_i):
                ret = cap.grab()

            ret, frame = cap.retrieve()
            if not ret:
                break
            frames.append(frame)

            dist = imagedist(frame * mask.detect, mask.orig, mask.detect, mask.orig_pixelcount)
            dists.append(dist)

            #print(dist)

            prev_i = i
            fi += delta_secs * framerate
            i = int(fi)
        return frames, dists

    def _get_threshold(self, dists):
        meandist = np.mean(dists)
        mindist = np.min(dists)
        threshold = mindist + (meandist - mindist) * 0.5
        return threshold

    def _get_asoc_frames(self, frames, dists, threshold):
        asocs = []
        prev_dist = 500.0
        count = 0
        for i, dist in reversed(list(enumerate(dists))):
            if dist < threshold and prev_dist >= threshold:
                count += 1
                frame = frames[i]
                asocs.append(frame)
                if count > 1:
                    break
            prev_dist = dist
        return list(reversed(asocs))

    def _process_asoc_frames(self, asocs, mask):
        words = {"1":{}, "2":{}}
        images = {"1":{}, "2":{}}
        for mask_key in mask.fields:
            for i, asoc in enumerate(asocs):
                cropped = mask.cropped_field(asoc, mask_key)
                processed = process_image(cropped)
                #showframes(processed)
                images[str(i + 1)][mask_key] = processed
                words[str(i + 1)][mask_key] = im2str(processed).upper()
        return images, words

    def _make_dirs(self, code):
        outroot = r"out/"
        outdir = outroot + code + r"/"
        mkdir(outroot)
        mkdir(outdir)
        return outroot, outdir

    def _get_output_string(self, season, episode, date, url, words):
        assert type(season) == str
        assert type(episode) == str
        assert type(date) == str
        assert type(url) == str

        fmt = """%(season)s. циклус
%(episode)s. емисија
%(date)s
%(url)s


А1 - %(1A1)s
А2 - %(1A2)s
А3 - %(1A3)s
А4 - %(1A4)s
 А - %(1A)s
Б1 - %(1B1)s
Б2 - %(1B2)s
Б3 - %(1B3)s
Б4 - %(1B4)s
 Б - %(1B)s
В1 - %(1V1)s
В2 - %(1V2)s
В3 - %(1V3)s
В4 - %(1V4)s
 В - %(1V)s
Г1 - %(1G1)s
Г2 - %(1G2)s
Г3 - %(1G3)s
Г4 - %(1G4)s
 Г - %(1G)s
??? - %(1K)s

А1 - %(2A1)s
А2 - %(2A2)s
А3 - %(2A3)s
А4 - %(2A4)s
 А - %(2A)s
Б1 - %(2B1)s
Б2 - %(2B2)s
Б3 - %(2B3)s
Б4 - %(2B4)s
 Б - %(2B)s
В1 - %(2V1)s
В2 - %(2V2)s
В3 - %(2V3)s
В4 - %(2V4)s
 В - %(2V)s
Г1 - %(2G1)s
Г2 - %(2G2)s
Г3 - %(2G3)s
Г4 - %(2G4)s
 Г - %(2G)s
??? - %(2K)s
"""
        d = dict()
        d["season"] = season
        d["episode"] = episode
        d["date"] = date
        d["url"] = url
        for c in ["1", "2"]:
            for a in ["A", "B", "V", "G"]:
                for b in ["1", "2", "3", "4", ""]:
                    d[c + a + b] = ""
            d[c + "K"] = ""
        for k, ws in words.items():
            assert type(k) == str
            for f, w in ws.items():
                key = k + f

                #print(key)

                val = w
                d[key] = val
        outstr = fmt % d
        return outstr

    def _writeout_txt(self, outdir, code, season, episode, date, url, words):
        outstr = self._get_output_string(season, episode, date, url, words)
        txtfilepath = outdir + code + ".txt"
        with open(txtfilepath, 'w+') as f:
            f.write(outstr)

    def _writeout_asoc_frames(self, outdir, asocs):
        for i, asoc in enumerate(asocs):
            cv2.imwrite(outdir + r"asocijacija" + str(i + 1) + r".png", asoc)

    def _writeout_images(self, outdir, images):
        fmt = outdir + r"asocijacija" + "%s%s.png"
        for k, ims in images.items():
            assert type(k) == str
            for f, im in ims.items():
                impath = fmt % (k, f)

                #print(impath)

                cv2.imwrite(impath, im)


    def process_episode(self, filename, season, episode, date, url):
        print("Processing video file '%s'" % filename)
        season, episode, date, url = self._format_episode_params(season, episode, date, url)
        code = "c" + season + "e" + episode + "a"

        cap = cv2.VideoCapture(filename)
        width, height, total_frames, framerate = self._get_video_params(cap)

        print("  Processing frames")
        mask = self._get_mask(season, episode, width, height)
        frames, dists = self._process_frames(cap, total_frames, framerate, mask)
        #print(dists)

        print("  Detecting final frames for the association quiz segments")
        threshold = self._get_threshold(dists)
        asocs = self._get_asoc_frames(frames, dists, threshold)

        print("  Performing OCR")
        images, words = self._process_asoc_frames(asocs, mask)
        #print(words)

        print("  Outputting results")
        outroot, outdir = self._make_dirs(code)
        self._writeout_txt(outroot, code, season, episode, date, url, words)
        self._writeout_asoc_frames(outdir, asocs)
        self._writeout_images(outdir, images)

        print("  Done")



filenames = sys.argv[1:]
if len(filenames) == 0:
    print("No input files to process. Exiting...")
    sys.exit(0)

# Check if all input files exist
for fn in filenames:
    if not os.path.isfile(fn):
        print("error: file '%s' doesn't exist" % fn, file = sys.stderr)
        sys.exit(-2)
    filename = fn + ".info.json"
    if not os.path.isfile(filename):
        print("error: file '%s' doesn't exist" % filename, file = sys.stderr)
        sys.exit(-2)

ep = EpisodeProcessor()
for fn in filenames:
    filename = fn + ".info.json"
    info = get_info_dict(filename)
    ep.process_episode(fn, info["season"], info["episode"], info["date"], info["url"])
sys.exit(0)
