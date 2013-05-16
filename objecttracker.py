#!/usr/bin/env python
import cv
import Image
import logging
from optparse import OptionParser
from traceback import format_exc

from util import compute_time

MAIN_WINDOW = "ObjectTracker"


def is_rect_within_img(rect, img):
    x, y, w, h = rect
    return 0 <= x < img.width and 0 <= y < img.height and 0 <= x + w < img.width and 0 <= y + h < img.height

def is_rect_nonzero(r):
    (_,_,w,h) = r
    return (w > 0) and (h > 0)


@compute_time
def apply_overlay_image(src, overlay, (x, y)):
    """src: Image from OpenCV, overlay: Image from PIL"""
    x, y = map(int, (x, y))
    if not (0 <= x < src.width and 0 <= y < src.height):
        return src
    img = Image.fromstring("RGB", cv.GetSize(src), src.tostring()[::-1]).rotate(180).convert("RGBA")
    img.paste(overlay, (x, y), overlay)

    cv_im = cv.CreateImageHeader(img.size, cv.IPL_DEPTH_8U, 3)
    cv.SetData(cv_im, img.convert("RGB").rotate(180).tostring()[::-1], img.size[0]*3)
    #cv.CvtColor(sub, sub, cv.CV_RGB2BGR)
    return cv_im


class ObjectTracker:
    def __init__(self, window_name, overlay):
        cv.NamedWindow(window_name, cv.CV_WINDOW_AUTOSIZE)
        cv.SetMouseCallback(window_name, self.mouse_handler)
        self.window_name = window_name
        self.overlay = Image.open(overlay).convert("RGBA")
        w, h = map(float, self.overlay.size)
        self.overlay_aspect_ratio = w / h
        self.reset()

    def reset(self):
        self.hist = cv.CreateHist([180], cv.CV_HIST_ARRAY, [(0, 180)], 1)
        self.drag_start = None      # Set to (x,y) when mouse starts drag
        self.track_window = None    # Set to rect when the mouse drag finishes
        self.selection = None

    def mouse_handler(self, event, x, y, flags, param):
        if event == cv.CV_EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
        if event == cv.CV_EVENT_LBUTTONUP:
            self.drag_start = None
            self.track_window = self.selection
        if self.drag_start:
            xmin = min(x, self.drag_start[0])
            ymin = min(y, self.drag_start[1])
            xmax = max(x, self.drag_start[0])
            ymax = max(y, self.drag_start[1])
            self.selection = (xmin, ymin, xmax - xmin, ymax - ymin)

    def detect_target(self, img, hue):
        sub = cv.GetSubRect(img, self.selection)
        save = cv.CloneMat(sub)
        cv.ConvertScale(img, img, 0.5)
        cv.Copy(save, sub)
        x, y, w, h = self.selection
        cv.Rectangle(img, (x, y), (x + w, y + h), (255, 255, 255))

        sel = cv.GetSubRect(hue, self.selection)
        cv.CalcArrHist([sel], self.hist, 0)
        (_, max_val, _, _) = cv.GetMinMaxHistValue(self.hist)
        if max_val != 0:
            cv.ConvertScale(self.hist.bins, self.hist.bins, 255. / max_val)

    def cam_shift(self, img, hue):
        # Compute back projection
        backproject = cv.CreateImage(cv.GetSize(img), 8, 1)

        # Run the cam-shift
        cv.CalcArrBackProject([hue], backproject, self.hist)
        crit = (cv.CV_TERMCRIT_EPS | cv.CV_TERMCRIT_ITER, 10, 1)
        (iters, (area, value, rect), track_box) = cv.CamShift(backproject, self.track_window, crit)
        self.track_window = rect
        return track_box

    def overlayed(self, img, track_box):
        #cv.EllipseBox(img, track_box, cv.CV_RGB(45, 255, 45), -1, cv.CV_AA, 0)
        maxdim = max(track_box[1])
        w, h = map(int, (maxdim * self.overlay_aspect_ratio, maxdim))
        overlay = self.overlay.resize((w, h), Image.ANTIALIAS)
        top_left = (track_box[0][0] - w / 2, track_box[0][1] - h / 2)
        return apply_overlay_image(img, overlay, top_left)

    def track_object(self, img):
        # Convert to HSV and keep the hue
        hsv = cv.CreateImage(cv.GetSize(img), 8, 3)
        cv.CvtColor(img, hsv, cv.CV_BGR2HSV)
        hue = cv.CreateImage(cv.GetSize(img), 8, 1)
        cv.Split(hsv, hue, None, None, None)

        # If mouse is pressed, highlight the current selected rectangle
        # and recompute the histogram
        if self.drag_start and is_rect_nonzero(self.selection):
            self.detect_target(img, hue)

        elif self.track_window and is_rect_nonzero(self.track_window):
            track_box = self.cam_shift(img, hue)
            if is_rect_nonzero(self.track_window) and is_rect_within_img(self.track_window, img):
                img = self.overlayed(img, track_box)
        return img

    def mainloop(self):
        capture = cv.CaptureFromCAM(0)
        while True:
            try:
                frame = cv.QueryFrame(capture)
                # Mirror
                cv.Flip(frame, frame, 1)
                img = self.track_object(frame)
                cv.ShowImage(self.window_name, img)
                if cv.WaitKey(10) >= 0:
                    break
            except:
                logging.critical(format_exc())
                self.reset()
        cv.DestroyWindow(self.window_name)


def main():
    parser = OptionParser(usage="usage: %prog [options]")
    parser.add_option("-o", "--overlay", action="store", dest="overlay", type="str",
                      help="Object tracking overlay image",
                      default="images/python.png")
    (options, args) = parser.parse_args()

    tracker = ObjectTracker(MAIN_WINDOW, options.overlay)
    tracker.mainloop()


if __name__ == '__main__':
    main()
