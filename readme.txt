        ****************
        **  CaraCara  **
        ****************

OpenCV demo by Rodolfo Carvalho.

Requirements:

    * OpenCV and Python bindings
    * PIL
    * A webcam


Usage example:

    1) Run `python caracara.py`.
    2) Place your face in a visible spot.
    3) You should have a balloon with text near your head.
    4) Drag the mouse over a region to track an object in that region.
       Use for example a colorful object easily distinguishable from
       the background.
    5) You should see an image over your tracked object.
    6) Try to move the object and the overlay image will follow.
    7) Press any key to exit.


Alternatives:

    * To use a different overlay image:
        `python caracara.py -o images/tux.png`

    * Read from an image instead of a webcam:
        `python caracara.py -f images/family-600x480.jpg`

    * Use objecttracker.py standalone:
        objecttracker.py is a part of caracara.py and can be used on its own.
        To track objects:
            `python objecttracker.py -o images/tux.png`
        Drag over a region with the mouse.
