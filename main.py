# Pycharm issue with imports. https://stackoverflow.com/questions/23248017/cannot-find-reference-xxx-in-init-py-python-pycharm
# from . import cv2
# __all__ = [cv2]
import cv2
import numpy as np


def convert_world_to_cam(point_array):
    # print("Converting " + str(point_array) + " to camera coords...")
    world_coords = np.array([[point_array[0]], [point_array[1]], [point_array[2]]])
    transformed = rotation_matrix * world_coords + translation_vector
    transformed = transformed.tolist()
    u = fx * (transformed[0][0] / transformed[2][0]) + cx
    v = fy * (transformed[1][0] / transformed[2][0]) + cy
    # print("u: " + str(u))
    # print("v: " + str(v))
    # print("---")
    return [u, v]

def return_route_array():
    # scale = 1000000
    scale = 110550039
    source_data = [[43.25696, -79.92581, 0], [43.25756, -79.92589, 0], [43.25757, -79.92562, 0],
                   [43.25762, -79.9238, 0], [43.25767, -79.92288, 0], [43.25764, -79.92281, 0],
                   [43.25766, -79.92177, 0], [43.25769, -79.92098, 0], [43.25773, -79.91845, 0],
                   [43.2578, -79.91574, 0], [43.25787, -79.91307, 0], [43.25791, -79.91211, 0],
                   [43.25792, -79.91161, 0], [43.25794, -79.91153, 0], [43.25801, -79.9114, 0],
                   [43.25804, -79.91111, 0], [43.25812, -79.9104, 0], [43.2585, -79.90693, 0], [43.25869, -79.9055, 0],
                   [43.25884, -79.90441, 0], [43.25901, -79.90327, 0], [43.25913, -79.90239, 0],
                   [43.2593, -79.90237, 0], [43.25965, -79.90225, 0], [43.26039, -79.90189, 0],
                   [43.26301, -79.90074, 0], [43.26309, -79.90106, 0]]

    car_location = [-79.92589, 43.25756, 0]

    offset = [0, 0, 0]
    data = []
    for i in range(0, len(source_data)):
        data.append([source_data[i][1] + offset[1], source_data[i][0] + offset[0],
                     source_data[i][2] + offset[2]])
    print(data)
    route_gps = np.array(data, dtype=np.float32)

    print(route_gps)
    car_gps = np.array(car_location, dtype=np.float32)

    # subtract car's current position
    relative_route_gps = (route_gps - car_gps)*scale
    relative_route_gps = relative_route_gps.tolist()
    print(relative_route_gps)
    print("--------")
    print(np.sqrt(pow(relative_route_gps[1][1],2)+pow(relative_route_gps[1][2],2))/1000)
    print("--------")
    return relative_route_gps


def nothing(x):
    pass


route_overview = return_route_array()


img = cv2.imread('5s.jpg', 1) #single image
img2 = cv2.imread('5s.jpg', 1) #single image

# intrinsic parameters (iPhone 5s)
fx = 2797.43
fy = 2797.43
cx = 1631.5
cy = 1223.5

# extrinsic parameters
x_theta = 1.57
y_theta = 0
z_theta = 0.771
translation_vector = np.array([[0], [0], [1500]], np.int32)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 1000, 1000)

cv2.createTrackbar('rx', 'image', 7912, 6280*2, nothing)
cv2.createTrackbar('ry', 'image', 0, 6280*2, nothing)
cv2.createTrackbar('rz', 'image', 6280, 6280*2, nothing)
cv2.createTrackbar('tx', 'image', 0, 10000, nothing)
cv2.createTrackbar('ty', 'image', 75, 10000, nothing)
cv2.createTrackbar('tz', 'image', 175, 10000, nothing)

while True:
    x_rotation_matrix = np.matrix(
        [[1, 0, 0], [0, np.cos(x_theta), -np.sin(x_theta)], [0, np.sin(x_theta), np.cos(x_theta)]])
    y_rotation_matrix = np.matrix(
        [[np.cos(y_theta), 0, np.sin(y_theta)], [0, 1, 0], [-np.sin(y_theta), 0, np.cos(y_theta)]])
    z_rotation_matrix = np.matrix(
        [[np.cos(z_theta), -np.sin(z_theta), 0], [np.sin(z_theta), np.cos(z_theta), 0], [0, 0, 1]])

    # rotation_matrix = x_rotation_matrix * y_rotation_matrix * z_rotation_matrix
    rotation_matrix = z_rotation_matrix * y_rotation_matrix * x_rotation_matrix

    transformed_points = []

    for i in range(0, len(route_overview)):
        transformed_points.append(convert_world_to_cam(route_overview[i]))

    imgimg = np.zeros((2448, 3264, 3), np.uint8)

    pts_world = np.array(transformed_points, np.int32)

    cv2.polylines(imgimg, [pts_world], False, (0, 0, 255), 15)

    cv2.imshow("image", imgimg)
    # cv2.imshow("image", img)

    x_theta = cv2.getTrackbarPos('rx', 'image')/1000.0-3.14
    y_theta = cv2.getTrackbarPos('ry', 'image')/1000.0-3.14
    z_theta = cv2.getTrackbarPos('rz', 'image')/1000.0-3.14

    translation_vector = np.array([[cv2.getTrackbarPos('tx', 'image')], [cv2.getTrackbarPos('ty', 'image')], [cv2.getTrackbarPos('tz', 'image')]], np.int32)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()