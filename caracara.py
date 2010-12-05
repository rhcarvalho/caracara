#!/usr/bin/python
"""
This program is demonstration for face and object detection using haar-like features.
The program finds faces in a camera image or video stream and displays a red box around them.

Original C implementation by:  ?
Python implementation by: Roman Stanchak, James Bowman
"""
import sys
import cv
from optparse import OptionParser
from functools import partial
from random import randint

# TODO:
# - Integrate with camshift.py
# - Put sprite on top of tracked object
# - Put bubble thoughts near faces


# Parameters for haar detection
# From the API:
# The default parameters (scale_factor=2, min_neighbors=3, flags=0) are tuned
# for accurate yet slow object detection. For a faster operation on real video
# images the settings are:
# scale_factor=1.2, min_neighbors=2, flags=CV_HAAR_DO_CANNY_PRUNING,
# min_size=<minimum possible face size

min_size = (20, 20)
image_scale = 2
haar_scale = 1.2
min_neighbors = 2
haar_flags = 0
MAIN_WINDOW = "CaraCara"


def detect_faces(img, cascade):
    # allocate temporary images
    gray = cv.CreateImage((img.width,img.height), 8, 1)
    small_img = cv.CreateImage((cv.Round(img.width / image_scale),
                   cv.Round (img.height / image_scale)), 8, 1)

    # convert color input image to grayscale
    cv.CvtColor(img, gray, cv.CV_BGR2GRAY)

    # scale input image for faster processing
    cv.Resize(gray, small_img, cv.CV_INTER_LINEAR)

    cv.EqualizeHist(small_img, small_img)

    t = cv.GetTickCount()
    faces = cv.HaarDetectObjects(small_img, cascade, cv.CreateMemStorage(0),
                                 haar_scale, min_neighbors, haar_flags, min_size)
    t = cv.GetTickCount() - t
    print "detection time = %gms" % (t/(cv.GetTickFrequency()*1000.))
    
    # the input to cv.HaarDetectObjects was resized, so scale the
    # bounding box of each face
    scaled_faces = [tuple(map(lambda k: int(k * image_scale), vec)) for (vec, n) in faces]
    return scaled_faces


def draw_surrounding_rectangles(img, faces):
    for (x, y, w, h) in faces:
        pt1 = (x, y)
        pt2 = (x + w, y + h)
        dark_violet = cv.RGB(148, 0, 211)
        cv.Rectangle(img, pt1, pt2, dark_violet, 1, 8, 0)


def capture_from_webcam(index):
    capture = cv.CreateCameraCapture(index)
    frame_copy = None
    while True:
        frame = cv.QueryFrame(capture)
        if not frame:
            cv.WaitKey(0)
            break
        if frame_copy is None:
            frame_copy = cv.CreateImage((frame.width, frame.height),
                                        cv.IPL_DEPTH_8U, frame.nChannels)
        if frame.origin == cv.IPL_ORIGIN_TL:
            cv.Copy(frame, frame_copy)
        else:
            cv.Flip(frame, frame_copy, 0)

        yield frame_copy


def capture_from_file(file):
    frame = cv.LoadImage(file, 1)
    frame_copy = None
    while True:
        if frame_copy is None:
            frame_copy = cv.CreateImage((frame.width, frame.height),
                                        cv.IPL_DEPTH_8U, frame.nChannels)
        cv.Copy(frame, frame_copy)
        yield frame_copy


def write_text(img, text, faces, color=cv.RGB(0, 0, 0)):
    font = cv.InitFont(fontFace=cv.CV_FONT_HERSHEY_PLAIN, hscale=1.0, vscale=1.0, shear=0, thickness=1, lineType=cv.CV_AA)
    for (x, y, w, h) in faces[:1]:
        origin = (x - 130 + randint(-3, 3), y - 35 + randint(-3, 3))
        cv.PutText(img, text, origin, font, color)


def draw_baloons(img, faces, color=cv.RGB(255, 255, 255)):
    for (x, y, w, h) in faces[:1]:
        triangle = ((x - 50, y - 20), (x + 10, y - 20), (x + randint(-10, 10), y))
        cv.FillConvexPoly(img, triangle, color=color, lineType=cv.CV_AA, shift=0)
        #cv.Circle(img, center=(130, 20), radius=10, color=color, thickness=-1, lineType=cv.CV_AA, shift=0)
        cv.EllipseBox(img, box=((x - 60, y - 40), (280, 50), 2), color=color, thickness=-1, lineType=cv.CV_AA, shift=0)


def main():
    parser = OptionParser(usage = "usage: %prog [options] [camera_index]")
    parser.add_option("-c", "--cascade", action="store", dest="cascade", type="str",
                      help="Haar cascade file, default %default",
                      default="cascades/haarcascade_frontalface_alt.xml")
    parser.add_option("-f", "--file", action="store", dest="file", type="str",
                      help="Image file")
    (options, args) = parser.parse_args()

    cascade = cv.Load(options.cascade)

    cv.NamedWindow(MAIN_WINDOW, cv.CV_WINDOW_AUTOSIZE)

    if options.file:
        image_iterator = capture_from_file(options.file)
    else:
        index = args[:1] and args[0].isdigit() and int(args[0]) or 0
        image_iterator = capture_from_webcam(index)

    for img in image_iterator:
        faces = detect_faces(img, cascade)
        draw_surrounding_rectangles(img, faces)
        draw_baloons(img, faces)
        write_text(img, "Go go my script!", faces)
        cv.ShowImage(MAIN_WINDOW, img)
        if cv.WaitKey(10) >= 0:
            break

    cv.DestroyWindow(MAIN_WINDOW)


if __name__ == '__main__':
    main()
