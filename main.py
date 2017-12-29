# Notes/Ideas:
# - Subdivide the array and then look at the altitudes for all of the points
#   Delete points in the array which don't have a large difference in altitude, simplifying the array
#   Will be efficient at runtime and give us altitudes accurately
#
# - I predict that we will encounter playback issues with regards to the frame rate
#   WaitKey will wait a certain amount of time before advancing to the next loop of the While-loop
#   We will experience a ms delay with running this code in addition to the predefined WaitKey delay
#   This will make it difficult to playback at 30 FPS consistently and have the video played at its true speed
#   Currently there is a ~22 ms delay from running this code, if this exceeds 33.3 ms we may run into issues playing back video at 30 FPS (1/30*1000 = 33.3 ms)
#   I discovered that threading in Python is terrible. In most cases, adding multiple threads is SLOWER than performing everything in the same thread
#   We should separate the program into 2 Processes: one for playing back the video, and the other for performing the math operations
#   Optionally we can have a third process for the Image Processing portion of OpenCV
#   We would have to research how to make these processes communicate with each other since their variables are all localized

import cv2
import numpy as np
import time
import requests
import polyline
import json


def point_is_viewable(line_pt_1, line_pt_2, point):
    # Determinant of form: (Bx - Ax) * (Y - Ay) - (By - Ay) * (X - Ax)
    # The 'line' is defined by two points. This is the clipping line.
    # This line is perpendicular to the direction the camera is facing.
    # The sign of the determinant essentially tells you which side of the line the 'point' is on
    # Using this property, we can determine if a point lies in front or behind the camera
    det = (line_pt_2[0]-line_pt_1[0])*(point[1]-line_pt_1[1])-(line_pt_2[1]-line_pt_1[1])*(point[0]-line_pt_1[0])
    if det < 1:
        return False
    else:
        return True


def get_intersection(line, P3, P4):
    # 'line' is an infinitely long clipping line. It is inputted into the function as an array of the Standard Form Coefficients [A, B, C]
    # P3 and P4 define a line segment

    # creating lines in Standard Form
    A2 = P4[1] - P3[1]
    B2 = P3[0] - P4[0]
    C2 = A2 * P3[0] + B2 * P3[1]

    # if det == 0, lines are parallel
    # det = A1 * B2 - A2 * B1
    det = line[0] * B2 - A2 * line[1]

    if det != 0:
        x = (B2 * line[2] - line[1] * C2) / det
        y = (line[0] * C2 - A2 * line[2]) / det
    else:
        x = 0
        y = 0

    return [x, y, 0]


def clip_array(route_array, clip_point, clip_rotation):
    # the clipping line is a line defined
    # right clip is a boolean indicating that this function will return the right bound clip. Pair this function with one that clips the left bound.
    route_array = route_array.tolist()

    clip_rotation -= np.pi/2

    # First clip line
    P1 = clip_point
    P2 = clip_point + np.array([np.cos(clip_rotation), np.sin(clip_rotation), 0])

    A1 = P2[1] - P1[1]
    B1 = P1[0] - P2[0]
    C1 = A1 * P1[0] + B1 * P1[1]

    clipped_array = []

    try:
        prev_point_viewable = point_is_viewable(P1, P2, route_array[0])

        if prev_point_viewable:
            clipped_array.append(route_array[0])

        for i in range(1, len(route_array)):
            viewable = point_is_viewable(P1, P2, route_array[i])

            if viewable != prev_point_viewable:  # add intersection point to the array
                intersection = get_intersection([A1, B1, C1], route_array[i], route_array[i - 1])
                clipped_array.append(intersection)

            if viewable:  # only append viewable points to the main output array
                clipped_array.append(route_array[i])

            prev_point_viewable = viewable

    except IndexError:
        print("Error")
        return np.array(get_relative_route(global_route, scale))  # if something goes wrong, just return the array which was originally provided

    return np.array(clipped_array)


def draw_road(route_array, width):
    incident_array = []
    angle_at_vertex = []
    angle_array = []
    magnitude_array = []
    road_array = []

    # calculate incident angle of each segment
    for i in range(0, len(route_array)-1):
        vector = route_array[i+1] - route_array[i]

        # Arctan is not very good at getting the desired angle. These if-statements make any corrections.
        if vector[1] != 0:  # Handles division by zero
            if vector[0] < 0 and vector[1] < 0:
                angle = np.arctan(vector[1] / vector[0]) - np.pi
            else:
                if vector[0] == 0:
                    if vector[1] > 0:
                        angle = np.pi/2
                    else:
                        angle = 3*np.pi/2
                else:
                    if vector[0] < 0 and vector[1] > 0:  # top left quadrant
                        angle = np.arctan(vector[1] / vector[0])+np.pi
                    else:
                        angle = np.arctan(vector[1] / vector[0])
        elif vector[0] > 0:
            angle = 0
        else:
            angle = np.pi

        if angle < 0:
            angle = 2*np.pi + angle
        incident_array.append(angle)

    # calculate angle at each vertex
    for i in range(0, len(route_array)-2):
        p_a = np.sqrt(np.square(route_array[i][0] - route_array[i + 1][0]) + np.square(route_array[i][1] - route_array[i + 1][1]))
        p_b = np.sqrt(np.square(route_array[i + 1][0] - route_array[i + 2][0]) + np.square(route_array[i + 1][1] - route_array[i + 2][1]))
        p_c = np.sqrt(np.square(route_array[i][0] - route_array[i + 2][0]) + np.square(route_array[i][1] - route_array[i + 2][1]))
        angle = np.arccos((np.square(p_a) + np.square(p_b) - np.square(p_c)) / (2 * p_a * p_b))
        angle_at_vertex.append(angle)

    # calculate required output angle at each node
    for i in range(0, len(incident_array)):

        # determine direction of road curvature (CW/CCW)
        if incident_array[i] >= np.pi:
            current_angle = -(2*np.pi - incident_array[i])
        else:
            current_angle = incident_array[i]

        if incident_array[i-1] >= np.pi:
            previous_angle = -(2*np.pi - incident_array[i-1])
        else:
            previous_angle = incident_array[i-1]

        if current_angle < previous_angle:
            clockwise = True
        else:
            clockwise = False

        # main angle calculation
        if i == 0:  # start point
            angle = incident_array[i] - np.pi/2
        else:
            if clockwise:
                angle = incident_array[i] - angle_at_vertex[i - 1]/2
            else:
                angle = incident_array[i] - np.pi + angle_at_vertex[i - 1]/2

        if angle < 0:
            angle = 2*np.pi + angle  # make all angles positive. this simplifies the rest of the algorithm.

        angle_array.append(angle)

    # end point
    angle_array.append(incident_array[-1] - np.pi / 2)

    for i in range(0, len(incident_array)):
        if i == 0:
            magnitude = width  # start/end point width is just the default width
        else:
            # print("incident (i-1): " + str(np.rad2deg(incident_array[i-1])))
            angle = incident_array[i-1] - np.pi/2
            if angle < 0:
                angle = 2*np.pi + angle

            magnitude = width / np.cos(np.abs(angle - angle_array[i]))

        magnitude_array.append(magnitude)

    # end point
    magnitude_array.append(width)

    # generate output road array
    for i in range(0, len(route_array)):
        coordinate = route_array[i] + [magnitude_array[i]*np.cos(angle_array[i]), magnitude_array[i]*np.sin(angle_array[i]), 0]
        road_array.append(coordinate)

    for i in range(len(route_array)-1, -1, -1):
        road_array.append(route_array[i])

    return road_array


def get_altitude(lat, long):
    api_url = 'https://maps.googleapis.com/maps/api/elevation/json?locations={0},{1}&key{2}'.format(str(lat), str(long), api_key)
    output = requests.get(api_url)
    altitude = json.loads(output.text)["results"][0]["elevation"]
    return round(altitude, 4)


def fetch_route_google_api(start, end, fetch_altitudes):
    api_url = 'https://maps.googleapis.com/maps/api/directions/json?origin={0},{1}&destination={2},{3}&key{4}'.format(str(start[0]), str(start[1]), str(end[0]), str(end[1]), api_key)
    polyline_data = requests.get(api_url).json()
    route_polyline = polyline.decode(polyline_data["routes"][0]["overview_polyline"]["points"])

    route_instructions = []
    route_array = polyline_data["routes"][0]["legs"][0]["steps"]
    for i in range(0, len(route_array)):  # create instruction array (ie. Nav events at respective GPS coordinate)
        route_instructions.append([[route_array[i]["start_location"]["lng"], route_array[i]["start_location"]["lat"]]])
        try:
            route_instructions[i].append(route_array[i]["maneuver"])
        except KeyError:  # no action
            route_instructions[i].append("starting point")
        route_instructions[i].append(route_array[i]["html_instructions"])

    route_polyline_vector = []
    for i in range(0, len(route_polyline)):  # convert to list with altitudes (list of xyz points)
        if fetch_altitudes:
            route_polyline_vector.append(
                [route_polyline[i][1], route_polyline[i][0], get_altitude(route_polyline[i][0], route_polyline[i][1])])  # fetch altitudes from Google API
        else:
            route_polyline_vector.append([route_polyline[i][1], route_polyline[i][0], 0])

    return np.array(route_polyline_vector, dtype=np.float32)  # return numpy array


def get_relative_route(array, scale):  # returns the route relative to car's current location. Also scale to relate GPS units to mm
    point = np.array(car_location, dtype=np.float32)
    relative_route_gps = (array - point)*scale  # subtract point from every element of array and adjust scale
    return relative_route_gps


def convert_world_to_cam(point_array):
    world_coords = np.array([[point_array[0]], [point_array[1]], [point_array[2]]])
    transformed = rotation_matrix * world_coords + translation_vector
    transformed = transformed.tolist()
    u = intrinsic_parameters[0] * (transformed[0][0] / transformed[2][0]) + intrinsic_parameters[2]
    v = intrinsic_parameters[1] * (transformed[1][0] / transformed[2][0]) + intrinsic_parameters[3]
    return [u, v]


def nothing(x):
    pass


api_key = 'AIzaSyDHOw34O0k8qDJ-td0jJhmi7GskJVffY64'
intrinsic_parameters = [2797.43, 2797.43, 1631.5, 1223.5]  # fx, fy, cx, cy

origin = [43.256963, -79.925822]  # replace this with car's gps coordinates. Make an initialization function which samples the car's current location, then queries Google for the global route
destination = [43.259598, -79.923227]

# global_route = fetch_route_google_api(origin, destination, False)  # Numpy array of polyline data. Boolean arg for altitudes
global_route = [[-79.92581, 43.25696, 0], [-79.92589, 43.25756, 0], [-79.92562, 43.25757, 0], [-79.9238, 43.25762, 0], [-79.92288, 43.25767, 0], [-79.92281, 43.25764, 0], [-79.92177, 43.25766, 0], [-79.92098, 43.25769, 0], [-79.91845, 43.25773, 0], [-79.91574, 43.2578, 0], [-79.91307, 43.25787, 0], [-79.91211, 43.25791, 0], [-79.91161, 43.25792, 0], [-79.91153, 43.25794, 0], [-79.9114, 43.25801, 0], [-79.91111, 43.25804, 0], [-79.9104, 43.25812, 0], [-79.90693, 43.2585, 0], [-79.9055, 43.25869, 0], [-79.90441, 43.25884, 0], [-79.90327, 43.25901, 0], [-79.90239, 43.25913, 0], [-79.90237, 43.2593, 0], [-79.90225, 43.25965, 0], [-79.90189, 43.26039, 0], [-79.90074, 43.26301, 0], [-79.90106, 43.26309, 0]]

road_width = 0.00002
global_route = draw_road(np.array(global_route), road_width)  # global route remains unaltered for the remainder of the code.
scale = 10000000  # relates GPS units to millimeters. 110550039 was the calculated value, it should be correct.

f = open('serial_output.txt', 'r')
start_time = time.time() - 0.1
prev_time = 0

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 1100, 768)

# CAR PERSPECTIVE
cv2.createTrackbar('rx', 'image', 1571, 6280*2, nothing)
cv2.createTrackbar('ry', 'image', 4712, 6280*2, nothing)
cv2.createTrackbar('rz', 'image', 6280, 6280*2, nothing)
cv2.createTrackbar('tx', 'image', 0, 10000, nothing)
cv2.createTrackbar('ty', 'image', 200, 10000, nothing)
cv2.createTrackbar('tz', 'image', 350, 10000, nothing)

# Used for measuring efficiency
iteration = 0
total_runtime = 0

keyboard_speed = 0.0001  # WSAD keyboard directions. Movement speed
dx = 0  # delete these after. They are used just for testing
dy = 0  # delete these after. They are used just for testing

while True:
    # used for runtime efficiency
    iteration += 1
    t1 = time.clock()

    #  Delete all instances of dx and dy after you have GPS data to work with. Set them to zero Clipping Point will be at the origin. car_location will just be the desired GPS coordinates of the car.
    car_location = [-79.92589 + dx, 43.25756 + dy, 0]
    relative_route = get_relative_route(global_route, scale)  # !! this could be more efficient. Right now we are moving the world camera coordinates around the camera. We should move the camera around the world.

    # fetch recorded data from txt file --------------------------------------------------------
    # try:
    #     current_time = round(time.time() - start_time, 1)
    #     if current_time != prev_time:  # ------------------------------------------------(10Hz) Update serial data
    #         data = f.readline().split(",")
    #         location = [float(data[1]), float(data[2]), 0]
    #         car_location = location
    #         # print("system_time: " + str(current_time))
    #         relative_route = get_relative_route(global_route, scale)
    #         if current_time - float(data[0]) > 0:  # data_time is lagging, therefore advance read line f'n
    #             while current_time - float(data[0]) > 0:
    #                 # print("Advancing line until current time is found...")
    #                 data = f.readline().split(",")
    #                 # print("new_data_time: " + data[0])
    #
    #         direction = float(data[3])
    #         # print("location: " + str(location))
    #         # print("direction: " + str(direction))
    #         car_location = location
    #         # print("-----")
    #     prev_time = current_time * 1
    # except (ValueError, IndexError):
    #     f.close()
    #     # print("-----\nReached end of Recording...\n-----")

    # Compute rotation matrix ----------------------------------------------------------------------
    x_theta = cv2.getTrackbarPos('rx', 'image')/1000.0-3.14
    z_theta = cv2.getTrackbarPos('rz', 'image')/1000.0-3.14
    y_theta = cv2.getTrackbarPos('ry', 'image') / 1000.0  # y_theta will simply be replaced with the compass data

    x_rotation_matrix = np.matrix(
        [[1, 0, 0], [0, np.cos(x_theta), -np.sin(x_theta)], [0, np.sin(x_theta), np.cos(x_theta)]])
    y_rotation_matrix = np.matrix(
        [[np.cos(y_theta), 0, np.sin(y_theta)], [0, 1, 0], [-np.sin(y_theta), 0, np.cos(y_theta)]])
    z_rotation_matrix = np.matrix(
        [[np.cos(z_theta), -np.sin(z_theta), 0], [np.sin(z_theta), np.cos(z_theta), 0], [0, 0, 1]])

    rotation_matrix = z_rotation_matrix * y_rotation_matrix * x_rotation_matrix  # we can hard-code the resultant matrix after we're done testing. It will be more efficient at runtime.
    translation_vector = np.array([[-cv2.getTrackbarPos('tx', 'image')], [cv2.getTrackbarPos('ty', 'image')], [cv2.getTrackbarPos('tz', 'image')]], np.int32)  # this should be replaced with the car's GPS data

    # Clipping & Perspective Transform -------------------------------------------------------------
    transformed_points = []
    clip_rotation = -cv2.getTrackbarPos('ry', 'image') / 1000.0 - np.pi/2
    clip_point = np.array([dx, dy, 0])
    relative_route = clip_array(relative_route, clip_point, clip_rotation)

    for i in relative_route:
        transformed_points.append(convert_world_to_cam(i))

    car_location_perspective_transform = convert_world_to_cam(clip_point)  # Used to draw the car's location as a red dot

    # OpenCV Drawing Functions -----------------------------------------------------------------------
    black_background = np.zeros((2448, 3264, 3), np.uint8)  # clears frame with black background. This will be replaced each frame by the following video frame. Black background is just for testing until we get video footage.
    cv2.polylines(black_background, [np.array(transformed_points, np.int32)], True, (255, 255, 255), 5)
    cv2.circle(black_background, (int(car_location_perspective_transform[0]), int(car_location_perspective_transform[1])), 35, (0, 0, 255), -1)
    cv2.imshow("image", black_background)

    # Measuring Runtime Efficiency -------------------------------------------------------------------
    end = time.clock()
    total_runtime += end - t1
    average_time = total_runtime/iteration
    # print("Average Runtime: " + str(round(average_time*1000,1)) + " milliseconds ; (" + str(iteration) + ") iterations.")

    key = cv2.waitKey(30)

    if key == ord('w'):
        dx += keyboard_speed

    if key == ord('s'):
        dx -= keyboard_speed

    if key == ord('a'):
        dy += keyboard_speed

    if key == ord('d'):
        dy -= keyboard_speed

    if key == ord('q'):
        break

cv2.destroyAllWindows()