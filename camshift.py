#!/usr/bin/env python

import cv

def is_rect_nonzero(r):
    (_,_,w,h) = r
    return (w > 0) and (h > 0)
    

def on_mouse(event, x, y, flags, param):
    global drag_start, track_window, selection
    if event == cv.CV_EVENT_LBUTTONDOWN:
        drag_start = (x, y)
    if event == cv.CV_EVENT_LBUTTONUP:
        drag_start = None
        track_window = selection
    if drag_start:
        xmin = min(x, drag_start[0])
        ymin = min(y, drag_start[1])
        xmax = max(x, drag_start[0])
        ymax = max(y, drag_start[1])
        selection = (xmin, ymin, xmax - xmin, ymax - ymin)

def run():
    global drag_start, track_window, selection
    hist = cv.CreateHist([180], cv.CV_HIST_ARRAY, [(0, 180)], 1)
    while True:
        frame = cv.QueryFrame(capture)

        # Convert to HSV and keep the hue
        hsv = cv.CreateImage(cv.GetSize(frame), 8, 3)
        cv.CvtColor(frame, hsv, cv.CV_BGR2HSV)
        hue = cv.CreateImage(cv.GetSize(frame), 8, 1)
        cv.Split(hsv, hue, None, None, None)

        # Compute back projection
        backproject = cv.CreateImage(cv.GetSize(frame), 8, 1)

        # Run the cam-shift
        cv.CalcArrBackProject([hue], backproject, hist)
        if track_window and is_rect_nonzero(track_window):
            crit = (cv.CV_TERMCRIT_EPS | cv.CV_TERMCRIT_ITER, 10, 1)
            (iters, (area, value, rect), track_box) = cv.CamShift(backproject, track_window, crit)
            track_window = rect

        # If mouse is pressed, highlight the current selected rectangle
        # and recompute the histogram
        if drag_start and is_rect_nonzero(selection):
            sub = cv.GetSubRect(frame, selection)
            save = cv.CloneMat(sub)
            cv.ConvertScale(frame, frame, 0.5)
            cv.Copy(save, sub)
            x, y, w, h = selection
            cv.Rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255))

            sel = cv.GetSubRect(hue, selection)
            cv.CalcArrHist([sel], hist, 0)
            (_, max_val, _, _) = cv.GetMinMaxHistValue(hist)
            if max_val != 0:
                cv.ConvertScale(hist.bins, hist.bins, 255. / max_val)
        elif track_window and is_rect_nonzero(track_window):
            cv.EllipseBox(frame, track_box, cv.CV_RGB(255, 0, 0), 3, cv.CV_AA, 0)

        cv.ShowImage("CamShiftDemo", frame)

        if cv.WaitKey(10) >= 0:
            break
            
    cv.DestroyWindow("CamShiftDemo")

if __name__=="__main__":
    capture = cv.CaptureFromCAM(0)
    cv.NamedWindow( "CamShiftDemo", 1 )
    cv.SetMouseCallback( "CamShiftDemo", on_mouse)

    drag_start = None      # Set to (x,y) when mouse starts drag
    track_window = None    # Set to rect when the mouse drag finishes
    selection = None
    
    
    run()
