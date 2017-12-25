import cv2
import numpy as np
import time
import requests
import polyline
import json


api_key = 'AIzaSyDHOw34O0k8qDJ-td0jJhmi7GskJVffY64'

# intrinsic parameters (iPhone 5s)
intrinsic_parameters = [2797.43, 2797.43, 1631.5, 1223.5]  # fx, fy, cx, cy

# extrinsic parameters
x_theta = 1.57
y_theta = 0
z_theta = 0.771
translation_vector = np.array([[0], [0], [1500]], np.int32)

location = [0, 0]  # Car location
direction = 0  # Car direction


def draw_road(route_array, width):
    incident_array = []
    angle_at_vertex = []
    angle_array = []
    magnitude_array = []
    road_array = []

    # calculate incident angle of each segment
    for i in range(0, len(route_array)-1):
        vector = route_array[i+1] - route_array[i]
        print(vector*10000)

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
                    if (vector[0] < 0 and vector[1] > 0):  # top left quadrant
                        angle = np.arctan(vector[1] / vector[0])+np.pi
                    else:
                        angle = np.arctan(vector[1] / vector[0])
                    print("angle: " + str(np.rad2deg(angle)))
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
                print("CW")
                print("Incident i - 1: "+str(np.rad2deg(incident_array[i-1])))
                print("Incident i: " + str(np.rad2deg(incident_array[i])))
                print("Angle: " + str(np.rad2deg(angle)))
            else:
                angle = incident_array[i] - np.pi + angle_at_vertex[i - 1]/2
                print("CCW")
                print("Incident i - 1: " + str(np.rad2deg(incident_array[i-1])))
                print("Incident i: " + str(np.rad2deg(incident_array[i])))
                print("Angle: " + str(np.rad2deg(angle)))

        if angle < 0:
            angle = 2*np.pi + angle  # make all angles positive. this simplifies the rest of the algorithm.
            print("Angle (fix neg): " + str(np.rad2deg(angle)))

        angle_array.append(angle)
        print("--")

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

            # print("Perpendicular Incident: " + str(np.rad2deg(angle)))
            # print("Angle Array: " + str(np.rad2deg(angle_array[i])))
            # print(" ")
        magnitude_array.append(magnitude)

    # end point
    magnitude_array.append(width)

    # generate output road array
    for i in range(0, len(route_array)):
        print("--")
        print("ANGLE: " + str(np.rad2deg(angle_array[i])))
        print("X: "+str(np.cos(angle_array[i])))
        print("Y: "+str(np.sin(angle_array[i])))
        if i == 0:
            print(">>> i=0; Angle:" + str(np.rad2deg(angle_array[i])))
            print(route_array[i])
            coordinate = route_array[i] + [magnitude_array[i]*np.cos(angle_array[i]), magnitude_array[i]*np.sin(angle_array[i]), 0]
        elif i == 1:
            print(">>> i=1; Angle:" + str(np.rad2deg(angle_array[i])))
            print(route_array[i])
            coordinate = route_array[i] + [magnitude_array[i] * np.cos(angle_array[i]), magnitude_array[i] * np.sin(angle_array[i]), 0]
        else:
            coordinate = route_array[i] + [magnitude_array[i]*np.cos(angle_array[i]), magnitude_array[i]*np.sin(angle_array[i]), 0]
        print("--")
        road_array.append(coordinate)

    print("---------------ANGLE ARRAY")
    for i in angle_array:
        print(round(np.rad2deg(i), 1))
    print("---------------")

    print("---------------INCIDENT ARRAY")
    for i in incident_array:
        print(round(np.rad2deg(i), 1))
    print("---------------")

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

    # print("Polyline Array:")
    print(route_polyline_vector)

    return np.array(route_polyline_vector, dtype=np.float32)  # return numpy array


def get_relative_route(array):  # returns the route relative to car's current location. Also scale to relate GPS units to mm
    scale = 1000000
    # scale = 110550039 #relates GPS units to millimeters

    point = np.array(car_location, dtype=np.float32)
    relative_route_gps = (array - point)*scale  # subtract point from every element of array and adjust scale
    # relative_route_gps = relative_route_gps.tolist()  # return relative route as list. I think its preferable to return a numpy array instead. So change this
    return relative_route_gps


def convert_world_to_cam(point_array):
    # print("Converting " + str(point_array) + " to camera coordinates...")
    world_coords = np.array([[point_array[0]], [point_array[1]], [point_array[2]]])
    transformed = rotation_matrix * world_coords + translation_vector
    transformed = transformed.tolist()
    u = intrinsic_parameters[0] * (transformed[0][0] / transformed[2][0]) + intrinsic_parameters[2]
    v = intrinsic_parameters[1] * (transformed[1][0] / transformed[2][0]) + intrinsic_parameters[3]
    # print("u: " + str(u))
    # print("v: " + str(v))
    # print("---")
    return [u, v]


def nothing(x):
    pass


car_location = [0, 0, 0]  # default value
origin = [43.256963, -79.925822]  # replace this with car's gps coordinate
destination = [43.263071, -79.901068]
# global_route = fetch_route_google_api(origin, destination, False)  # Numpy array of polyline data. Boolean arg for altitudes
global_route = [[-79.92581, 43.25696, 0], [-79.92589, 43.25756, 0], [-79.92562, 43.25757, 0], [-79.9238, 43.25762, 0], [-79.92288, 43.25767, 0], [-79.92281, 43.25764, 0], [-79.92177, 43.25766, 0], [-79.92098, 43.25769, 0], [-79.91845, 43.25773, 0], [-79.91574, 43.2578, 0], [-79.91307, 43.25787, 0], [-79.91211, 43.25791, 0], [-79.91161, 43.25792, 0], [-79.91153, 43.25794, 0], [-79.9114, 43.25801, 0], [-79.91111, 43.25804, 0], [-79.9104, 43.25812, 0], [-79.90693, 43.2585, 0], [-79.9055, 43.25869, 0], [-79.90441, 43.25884, 0], [-79.90327, 43.25901, 0], [-79.90239, 43.25913, 0], [-79.90237, 43.2593, 0], [-79.90225, 43.25965, 0], [-79.90189, 43.26039, 0], [-79.90074, 43.26301, 0], [-79.90106, 43.26309, 0]]
global_route = draw_road(np.array(global_route), 0.00015)
relative_route = get_relative_route(global_route)


# draw_road(relative_route)
print("-----")
# draw_road(np.array([[0, 0, 0], [1, 1, 0], [2, 1, 0], [3, 2, 0], [2, 3, 0], [1, 3, 0], [1, 4, 0], [0, 3, 0], [1, 2, 0]]), 0.1)

print("-----")

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 1100, 768)

# OLD DEFAULTS
cv2.createTrackbar('rx', 'image', 7912, 6280*2, nothing)
cv2.createTrackbar('ry', 'image', 4592, 6280*2, nothing)
cv2.createTrackbar('rz', 'image', 6280, 6280*2, nothing)
cv2.createTrackbar('tx', 'image', 77, 10000, nothing)
cv2.createTrackbar('ty', 'image', 44, 10000, nothing)
cv2.createTrackbar('tz', 'image', 66, 10000, nothing)

# NEW DEFAULTS
# cv2.createTrackbar('rx', 'image', 0, 6280*2, nothing)
# cv2.createTrackbar('ry', 'image', 0, 6280*2, nothing)
# cv2.createTrackbar('rz', 'image', 3140, 6280*2, nothing)
# cv2.createTrackbar('tx', 'image', 0, 30000, nothing)
# cv2.createTrackbar('ty', 'image', 0, 10000, nothing)
# cv2.createTrackbar('tz', 'image', 3029, 30000, nothing)


f = open('serial_output.txt', 'r')
start_time = time.time() - 0.1
prev_time = 0

# y_theta = np.pi*1.3

while True:
    try:
        current_time = round(time.time() - start_time, 1)
        if current_time != prev_time:  # ------------------------------------------------(10Hz) Update serial data
            data = f.readline().split(",")
            location = [float(data[1]), float(data[2]), 0]
            car_location = location
            # print("system_time: " + str(current_time))
            relative_route = get_relative_route(global_route)
            if current_time - float(data[0]) > 0:  # data_time is lagging, therefore advance read line f'n
                while current_time - float(data[0]) > 0:
                    # print("Advancing line until current time is found...")
                    data = f.readline().split(",")
                    # print("new_data_time: " + data[0])

            direction = float(data[3])
            # print("location: " + str(location))
            # print("direction: " + str(direction))
            car_location = location
            # print("-----")
        prev_time = current_time * 1
    except KeyboardInterrupt:
        f.close()
        # print("-----\nProgram Terminated by User...\n-----")
    except (ValueError, IndexError):
        f.close()
        # print("-----\nReached end of Recording...\n-----")

    # hard-coding car's location
    # car_location = [-79.92581, 43.25696, 0]

    x_rotation_matrix = np.matrix(
        [[1, 0, 0], [0, np.cos(x_theta), -np.sin(x_theta)], [0, np.sin(x_theta), np.cos(x_theta)]])
    y_rotation_matrix = np.matrix(
        [[np.cos(y_theta), 0, np.sin(y_theta)], [0, 1, 0], [-np.sin(y_theta), 0, np.cos(y_theta)]])
    z_rotation_matrix = np.matrix(
        [[np.cos(z_theta), -np.sin(z_theta), 0], [np.sin(z_theta), np.cos(z_theta), 0], [0, 0, 1]])

    rotation_matrix = z_rotation_matrix * y_rotation_matrix * x_rotation_matrix

    transformed_points = []

    for i in range(0, len(relative_route)):
        transformed_points.append(convert_world_to_cam(relative_route[i]))

    black_background = np.zeros((2448, 3264, 3), np.uint8)

    pts_world = np.array(transformed_points, np.int32)

    cv2.polylines(black_background, [pts_world], True, (255, 255, 0), 4)

    cv2.imshow("image", black_background)

    x_theta = cv2.getTrackbarPos('rx', 'image')/1000.0-3.14
    y_theta = cv2.getTrackbarPos('ry', 'image')/1000.0-2*3.14
    # y_theta = y_theta + np.pi*0.01
    z_theta = cv2.getTrackbarPos('rz', 'image')/1000.0-3.14

    # delete the '-300' offset on line below when you get the chance... also get rid of that negative tx.
    translation_vector = np.array([[-cv2.getTrackbarPos('tx', 'image')], [cv2.getTrackbarPos('ty', 'image')+11], [cv2.getTrackbarPos('tz', 'image')]], np.int32)

    if cv2.waitKey(60) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()