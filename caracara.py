#!/usr/bin/env python
import cv
import logging
from collections import deque
from optparse import OptionParser
from random import uniform
from traceback import format_exc

from objecttracker import ObjectTracker
from util import cached_times, compute_time

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


@cached_times(5)
@compute_time
def detect_faces(img, cascade):
    """Detect faces from img using cascade.

    Return a list of tuples (x, y, width, height) where (x, y) is
    the coordinate of the top left corner of a face."""
    # allocate temporary images
    gray = cv.CreateImage((img.width,img.height), 8, 1)
    small_img = cv.CreateImage((cv.Round(img.width / image_scale),
                   cv.Round (img.height / image_scale)), 8, 1)

    # convert color input image to grayscale
    cv.CvtColor(img, gray, cv.CV_BGR2GRAY)

    # scale input image for faster processing
    cv.Resize(gray, small_img, cv.CV_INTER_LINEAR)

    cv.EqualizeHist(small_img, small_img)

    faces = cv.HaarDetectObjects(small_img, cascade, cv.CreateMemStorage(0),
                                 haar_scale, min_neighbors, haar_flags, min_size)

    # the input to cv.HaarDetectObjects was resized, so scale the
    # bounding box of each face
    scaled_faces = [tuple(map(lambda k: int(k * image_scale), vec)) for (vec, n) in faces]
    return scaled_faces


def draw_surrounding_rectangles(img, faces):
    """Draw a rectangle around each face of img."""
    for (x, y, w, h) in faces:
        pt1 = (x, y)
        pt2 = (x + w, y + h)
        dark_violet = cv.RGB(148, 0, 211)
        cv.Rectangle(img, pt1, pt2, dark_violet, 1, 8, 0)


def capture_from_webcam(index):
    """Generator over webcam frames. Yields a new image infinitely."""
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
            #cv.Copy(frame, frame_copy)
            # Mirror
            cv.Flip(frame, frame_copy, 1)
        else:
            cv.Flip(frame, frame_copy, 0)

        yield frame_copy


def capture_from_file(file):
    """Generator over a single file. Yields a copy of the image file infinitely."""
    frame = cv.LoadImage(file, 1)
    frame_copy = None
    while True:
        if frame_copy is None:
            frame_copy = cv.CreateImage((frame.width, frame.height),
                                        cv.IPL_DEPTH_8U, frame.nChannels)
        cv.Copy(frame, frame_copy)
        yield frame_copy


def write_text(img, texts, faces, color=cv.RGB(0, 0, 0)):
    """Write a random text from texts next each face from faces of img.

    Length of texts must be greater than or equal length of faces."""
    font = cv.InitFont(fontFace=cv.CV_FONT_HERSHEY_PLAIN, hscale=1.0, vscale=1.0, shear=0, thickness=1, lineType=cv.CV_AA)
    for text, (x, y, w, h) in zip(texts, faces):
        # check size of rendered text
        (width, height), baseline = cv.GetTextSize(text, font)
        # bottom-left coordinates of text randomly shifted
        origin = (int((x - width) * uniform(0.95, 1.05)), int((y - height) * uniform(0.95, 1.05)))

        if width > img.width or height > img.height:
            logging.warning('Image is smaller than the text: (%d, %d) x (%d, %d)' % (img.width, img.height, width, height))
            break

        # test text boundaries against image
        if origin[0] < 0:
            logging.debug("Moved text balloon to the right")
            origin = (0, origin[1])

        if origin[0] + width > img.width:
            logging.debug("Moved text balloon to the left")
            origin = (img.width - width, origin[1])

        if origin[1] - height < 0:
            logging.debug("Moved text balloon down")
            origin = (origin[0], height)

        if origin[1] > img.height:
            logging.debug("Moved text balloon up")
            origin = (origin[0], img.height - baseline)

        center = (origin[0] + width / 2, origin[1] - height / 2)
        draw_balloon(img, (center[0], center[1], width, height))
        cv.PutText(img, text, origin, font, color)


def draw_balloon(img, rect, color=cv.RGB(255, 255, 255)):
    """Draw balloon centered in rect of img."""
    x, y, w, h = rect
    # pad width and height
    w, h = int(w * 1.4), int(h * 4.2)
    triangle = ((x, y), (x + w / 3, y), (int((x + w / 3) * uniform(0.95, 1.1)), y + h))
    cv.FillConvexPoly(img, triangle, color=color, lineType=cv.CV_AA, shift=0)
    angle = 2 * uniform(0, 1)
    cv.EllipseBox(img, box=((x, y), (w, h), angle), color=color, thickness=-1, lineType=cv.CV_AA, shift=0)



class CaraCara:
    def __init__(self, window_name, image_iterator, cascade, overlay):
        cv.NamedWindow(window_name, cv.CV_WINDOW_AUTOSIZE)
        self.window_name = window_name
        self.image_iterator = image_iterator
        self.cascade = cv.Load(cascade)
        self.texts = deque(("What am I gonna do?", "Mark an object with the mouse!", "Tracking..."))
        self.tracker = ObjectTracker(window_name, overlay)

    def mainloop(self):
        fps_buffer = []
        group_size = 20
        for img in self.image_iterator:
            try:
                t = cv.GetTickCount()
                faces = detect_faces(img, self.cascade)
                img = self.tracker.track_object(img)
                #draw_surrounding_rectangles(img, faces)
                write_text(img, self.texts, faces)
                cv.ShowImage(self.window_name, img)

                if cv.WaitKey(10) >= 0:
                    break

                t = cv.GetTickCount() - t
                fps_buffer.append((cv.GetTickFrequency() * 1000000.) / t)
                if len(fps_buffer) == group_size:
                    fps_buffer = [sum(fps_buffer) / group_size]
                    logging.info("%.4f fps" % fps_buffer[0])
                    self.texts.rotate(-1)
            except:
                logging.critical(format_exc())
                self.tracker.reset()
        cv.DestroyWindow(self.window_name)


def main():
    parser = OptionParser(usage="usage: %prog [options] [camera_index]")
    parser.add_option("-c", "--cascade", action="store", dest="cascade", type="str",
                      help="Haar cascade file, default %default",
                      default="cascades/haarcascade_frontalface_alt.xml")
    parser.add_option("-f", "--file", action="store", dest="file", type="str",
                      help="Image file")
    parser.add_option("-o", "--overlay", action="store", dest="overlay", type="str",
                      help="Object tracking overlay image",
                      default="images/python.png")
    (options, args) = parser.parse_args()

    if options.file:
        image_iterator = capture_from_file(options.file)
    else:
        index = args[:1] and args[0].isdigit() and int(args[0]) or 0
        image_iterator = capture_from_webcam(index)

    caracara = CaraCara(MAIN_WINDOW, image_iterator, options.cascade, options.overlay)
    caracara.mainloop()


if __name__ == '__main__':
    main()
