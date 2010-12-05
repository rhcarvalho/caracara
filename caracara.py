#!/usr/bin/python
"""
This program is demonstration for face and object detection using haar-like features.
The program finds faces in a camera image or video stream and displays a red box around them.

Original C implementation by:  ?
Python implementation by: Roman Stanchak, James Bowman
"""
import cv
import logging
import sys
from functools import partial
from optparse import OptionParser
from random import sample, uniform

from camshift import ObjectTracker

# TODO:
# - Put sprite on top of tracked object


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

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] %(message)s')


def cached_times(n):
    def decorator(func):
        memory = dict(
            call_counter = 0,
            cache = None
        )
        def cached_func(*args, **kwargs):
            if memory['call_counter'] % n == 0:
                memory['cache'] = func(*args, **kwargs)
            memory['call_counter'] = (memory['call_counter'] + 1) % n
            return memory['cache']
        return cached_func
    return decorator


@cached_times(5)
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

    t = cv.GetTickCount()
    faces = cv.HaarDetectObjects(small_img, cascade, cv.CreateMemStorage(0),
                                 haar_scale, min_neighbors, haar_flags, min_size)
    t = cv.GetTickCount() - t
    logging.info("detection time = %gms" % (t / (cv.GetTickFrequency() * 1000.)))

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
            cv.Copy(frame, frame_copy)
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
    # consider only faces for which we can pick up a text
    faces = faces[:len(texts)]
    texts = sample(texts, len(faces))
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
            logging.info("Moved text balloon to the right")
            origin = (0, origin[1])
        
        if origin[0] + width > img.width:
            logging.info("Moved text balloon to the left")
            origin = (img.width - width, origin[1])
            
        if origin[1] - height < 0:
            logging.info("Moved text balloon down")
            origin = (origin[0], height)
            
        if origin[1] > img.height:
            logging.info("Moved text balloon up")
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
    tracker = ObjectTracker(MAIN_WINDOW)

    if options.file:
        image_iterator = capture_from_file(options.file)
    else:
        index = args[:1] and args[0].isdigit() and int(args[0]) or 0
        image_iterator = capture_from_webcam(index)

    fps_buffer = []
    for img in image_iterator:
        t = cv.GetTickCount()
        faces = detect_faces(img, cascade)
        tracker.track_object(img)
        #draw_surrounding_rectangles(img, faces)
        texts = ("Go go my script!", "I am a hack3r :P", "OMG!")
        write_text(img, texts, faces)
        cv.ShowImage(MAIN_WINDOW, img)
        if cv.WaitKey(10) >= 0:
            break
        t = cv.GetTickCount() - t
        fps_buffer.append((cv.GetTickFrequency() * 1000000.) / t)
        if len(fps_buffer) == 10:
            fps_buffer = [sum(fps_buffer) / 10]
            logging.debug("%.4f fps" % fps_buffer[0])

    cv.DestroyWindow(MAIN_WINDOW)


if __name__ == '__main__':
    main()
