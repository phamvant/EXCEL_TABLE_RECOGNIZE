import cv2
import imutils
table_image = cv2.imwrite("")
table_origin = table_image.copy()

h, w , _ = table_origin.shape
tile = w // 1200

point = []
table_origin = imutils.resize(table_origin, width=1200)
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(table_origin, (x, y), 3, (0, 0, 255), 3, cv2.FILLED)
        x = x * tile
        y = y * tile
        point.append((x, y))
        cv2.imshow('Image', table_origin)
        if len(points) == 2:
            cv2.waitKey()
            return None


cv2.imshow('Image', table_origin)
points = []
cv2.setMouseCallback('Image', click_event)
cv2.waitKey()
cv2.destroyAllWindows()