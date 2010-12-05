#!/usr/bin/env python

import cv

def is_rect_nonzero(r):
    (_,_,w,h) = r
    return (w > 0) and (h > 0)


def overlay_image(src, overlay, location, S, D):
    location = tuple(map(int, location))
    for x in xrange(overlay.width):
        if x + location[0] >= src.width:
            continue
        for y in xrange(overlay.height):
            if y + location[1] >= src.height:
                continue
            source = cv.Get2D(src, y + location[1], x + location[0])
            over = cv.Get2D(overlay, y, x)
            merged = tuple(S[i] * source[i] + D[i] * over[i] for i in xrange(4))
            cv.Set2D(src, y + location[1], x + location[0], merged)


class ObjectTracker:
    def __init__(self, window):
        cv.SetMouseCallback(window, self.mouse_handler)
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


    def track_object(self, img):
        # Convert to HSV and keep the hue
        hsv = cv.CreateImage(cv.GetSize(img), 8, 3)
        cv.CvtColor(img, hsv, cv.CV_BGR2HSV)
        hue = cv.CreateImage(cv.GetSize(img), 8, 1)
        cv.Split(hsv, hue, None, None, None)

        # Compute back projection
        backproject = cv.CreateImage(cv.GetSize(img), 8, 1)

        # Run the cam-shift
        cv.CalcArrBackProject([hue], backproject, self.hist)
        if self.track_window and is_rect_nonzero(self.track_window):
            crit = (cv.CV_TERMCRIT_EPS | cv.CV_TERMCRIT_ITER, 10, 1)
            (iters, (area, value, rect), track_box) = cv.CamShift(backproject, self.track_window, crit)
            self.track_window = rect

        # If mouse is pressed, highlight the current selected rectangle
        # and recompute the histogram
        if self.drag_start and is_rect_nonzero(self.selection):
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
        elif self.track_window and is_rect_nonzero(self.track_window):
            #cv.EllipseBox(img, track_box, cv.CV_RGB(45, 255, 45), -1, cv.CV_AA, 0)
            overlay = cv.LoadImage("python.png")
            top_left = (track_box[0][0] - overlay.width / 2, track_box[0][1] - overlay.height / 2)
            overlay_image(img, overlay, top_left, (0,0,0,0), (1,1,1,1))


def main():
    MAIN_WINDOW = "ObjectTracker"
    capture = cv.CaptureFromCAM(0)
    cv.NamedWindow(MAIN_WINDOW, 1)
    tracker = ObjectTracker(MAIN_WINDOW)
    while True:
        frame = cv.QueryFrame(capture)
        tracker.track_object(frame)
        cv.ShowImage(MAIN_WINDOW, frame)
        if cv.WaitKey(10) >= 0:
            break
    cv.DestroyWindow(MAIN_WINDOW)


if __name__ == '__main__':
    main()
