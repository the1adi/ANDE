from multiprocessing import Process, Queue
import cv2
import numpy as np
import requests
import polyline
import json
import time
import pyclipper


class NavigationEvent:
    def __init__(self, txt, coordinates, is_drawable, is_viewable, is_turn):
        self.text_command = txt
        self.event_coordinates = coordinates
        self.relative_event_coordinates = coordinates
        self.event_screen_coordinates = (0, 0)
        self.is_drawable = is_drawable  # if desire to draw it or not
        self.is_viewable = is_viewable  # is the point clipped?
        self.is_turn = is_turn
        self.distance_from_vehicle = 0

    def get_dist(self, car_position):
        self.distance_from_vehicle = np.sqrt(np.square(car_position[0] - self.event_coordinates[0]) + np.square(car_position[1] - self.event_coordinates[1]))
        return self.distance_from_vehicle

    def set_relative_event_coordinates(self, coordinates):
        self.relative_event_coordinates = coordinates

    def set_event_screen_coordinates(self, coordinates):
        self.event_screen_coordinates = (int(coordinates[0]), int(coordinates[1]))

    def set_visibility(self, visibility):
        self.is_viewable = visibility


def gradient_blend(img1, img2, alpha):
    # alpha is the maximum opacity that the gradient will blend to
    output = np.zeros(img1.shape, np.uint8)
    height, width, depth = output.shape
    c = np.ones(int(height/2))[:, None, None]
    d = np.linspace(1, (1-alpha), int(height/2))[:, None, None]
    c = np.concatenate((c, d))
    output[0:height, 0:width, 0:depth] = c * img1[0:height, 0:width, 0:depth] + (1 - c)*img2[0:height, 0:width, 0:depth]
    return output


def intersection(l1, l2):
    d = l1[0] * l2[1] - l1[1] * l2[0]
    dx = l1[2] * l2[1] - l1[1] * l2[2]
    dy = l1[0] * l2[2] - l1[2] * l2[0]
    x = dx / d
    y = dy / d
    return x, y


def distance_squared(pt_1, pt_2):  # used for relative distances. There is no point to taking the square root, it will just increase computation time
    return np.square(pt_2[0]-pt_1[0]) + np.square(pt_2[1]-pt_1[1])


def point_is_on_line(pt, pt_1, pt_2, threshold):  # pt_1 and pt_2 define the line
    if distance_squared(pt_1, pt) + distance_squared(pt, pt_2) - distance_squared(pt_1, pt_2) < threshold:  # check if point is between other two points. Uses a threshold due to floats.
        return True
    else:
        return False


def get_closest_point(pt_1, pt_2, pt_3):
    # standard form coefficients (Ax + By = C)
    a1 = pt_2[1] - pt_1[1]
    b1 = pt_1[0] - pt_2[0]
    c1 = a1 * pt_1[0] + b1 * pt_1[1]

    a2 = b1
    b2 = -a1
    c2 = a2 * pt_3[0] + b2 * pt_3[1]

    intersection_pt = intersection([a1, b1, c1], [a2, b2, c2])

    # if distance_squared(pt_1, intersection_pt) + distance_squared(intersection_pt, pt_2) - distance_squared(pt_1, pt_2) < 0.0000001:  # check if point is between other two points. Uses a threshold due to floats.
    #     valid = True
    # else:
    #     valid = False

    if point_is_on_line(intersection_pt, pt_1, pt_2, 0.0000001):
        valid = True
    else:
        valid = False

    return intersection_pt, valid


def positive_angle(angle):
    # accepts degrees
    if angle < 0:
        return angle + 360
    else:
        return angle


def snap(pt, route, leg, rng):

    snapped_point, current_leg = ([0, 0], 0)

    for i in range(-rng, rng+1):
        try:
            if leg + i >= 0:  # stops the program from checking negative indices
                current_snapped_point, valid = get_closest_point(route[leg + i], route[leg + i + 1], pt)
                if valid and distance_squared(current_snapped_point, pt) < distance_squared(snapped_point, pt):
                    snapped_point = current_snapped_point
                    current_leg = leg + i
        except IndexError:
            pass

        for i in range(-rng, rng+1):
            try:
                if leg + i >= 0:
                    if distance_squared(snapped_point, pt) > distance_squared(route[leg + i], pt):
                        snapped_point = route[leg + i]
                        current_leg = leg + i
            except IndexError:
                pass

    # compute compass direction
    offset = 0
    direction_vector = route[current_leg + 1] - route[current_leg]
    direction = np.rad2deg(np.arctan(direction_vector[0]/direction_vector[1])) + offset

    # print("Snapped")
    # print(snapped_point)
    return snapped_point, current_leg, direction


def subdivide(route, scale):
    new_route = []

    for i in range(0, len(route)-1):
        new_route.append(route[i])  # always append first point in leg
        finished = False
        magnitude = np.sqrt(np.square(route[i+1][0] - route[i][0])+np.square(route[i+1][1] - route[i][1]))
        unit_vector = [(route[i+1][0] - route[i][0])/magnitude, (route[i+1][1] - route[i][1])/magnitude]
        # print(i)
        n = 1
        while not finished:
            # print(n)
            new_pt = [route[i][0] + unit_vector[0]*n*scale, route[i][1] + unit_vector[1]*n*scale]
            if point_is_on_line(new_pt, route[i], route[i + 1], 0.0000001):
                new_route.append(new_pt)
            else:
                finished = True
            n += 1

    return new_route


def rolling_average(new_data, average_array, roll_len):
    if len(average_array) >= roll_len:
        del average_array[0]
        average_array.append(new_data)
    else:
        for i in range(0, roll_len):
            average_array.append(new_data)
    # print(average_array)
    return np.average(average_array), average_array


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


def get_intersection(line, pt_3, pt_4):
    # 'line' is an infinitely long clipping line. It is inputted into the function as an array of the Standard Form Coefficients [A, B, C]
    # pt_3 and pt_4 define a line segment

    # creating lines in Standard Form
    a2 = pt_4[1] - pt_3[1]
    b2 = pt_3[0] - pt_4[0]
    c2 = a2 * pt_3[0] + b2 * pt_3[1]

    # if det == 0, lines are parallel
    # det = A1 * B2 - A2 * B1
    det = line[0] * b2 - a2 * line[1]

    if det != 0:
        x = (b2 * line[2] - line[1] * c2) / det
        y = (line[0] * c2 - a2 * line[2]) / det
    else:
        x = 0
        y = 0

    return [x, y, 0]


def clip_array(route_array, clip_point, clip_rotation, global_route, scale, segment):
    # segment is a boolean which specifies if the input array is a line/shape or if it is just an array of individual points which are not connected as a line
    # ie. segment = True would be used clip the road shape. This adds new points to the array when it clips things off the screen
    # segment = False would be used to clip the event triggers which are off the screen
    # segment = False returns a BOOLEAN which states for every point, whether it is visible on the screen or not

    route_array = route_array.tolist()

    clip_rotation -= np.pi/2

    # First clip line
    P1 = clip_point
    P2 = clip_point + np.array([np.cos(clip_rotation), np.sin(clip_rotation), 0])

    A1 = P2[1] - P1[1]
    B1 = P1[0] - P2[0]
    C1 = A1 * P1[0] + B1 * P1[1]

    clipped_array = []

    if segment:
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
    else:
        for i in route_array:
            clipped_array.append(point_is_viewable(P1, P2, i))

        return clipped_array

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


def draw_road_2(route, width, s):  # larger scalar will ensure there is less loss of data in the route. Road width must be sized relative to scalar.
    global_route = []
    for i in range(0, len(route)):  # scale GPS coordinates
        global_route.append([route[i][0] * s, route[i][1] * s])

    global_route_offset = pyclipper.PyclipperOffset()
    global_route_offset.AddPath(global_route, pyclipper.JT_SQUARE, pyclipper.ET_OPENROUND)
    global_route_expanded = global_route_offset.Execute(width)[0]  # expands road by defined road_width to the left/right of centre line

    output_array = []
    for coordinate in global_route_expanded:  # descale GPS coordinates
        output_array.append(np.array([coordinate[0]/s, coordinate[1]/s, 0]))

    return output_array


def get_altitude(key, lat, long):
    api_url = 'https://maps.googleapis.com/maps/api/elevation/json?locations={0},{1}&key{2}'.format(str(lat), str(long), key)
    output = requests.get(api_url)
    altitude = json.loads(output.text)["results"][0]["elevation"]
    return round(altitude, 4)


def fetch_route_google_api(key, start, end, fetch_altitudes):
    api_url = 'https://maps.googleapis.com/maps/api/directions/json?origin={0},{1}&destination={2},{3}&key{4}'.format(str(start[0]), str(start[1]), str(end[0]), str(end[1]), key)
    polyline_data = requests.get(api_url).json()

    # Sometimes Google API will reject the request due to lack of tokens. If you ask again it will usually accept the request.
    try:
        route_polyline = polyline.decode(polyline_data["routes"][0]["overview_polyline"]["points"])
    except IndexError:
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
                [route_polyline[i][1], route_polyline[i][0], get_altitude(key, route_polyline[i][0], route_polyline[i][1])])  # fetch altitudes from Google API
        else:
            route_polyline_vector.append([route_polyline[i][1], route_polyline[i][0], 0])

    navigation_event_array = []
    for event in route_instructions:
        gps_location = [event[0][0], event[0][1], 0]
        if event[1] == 'starting point':  # starting point isn't a command
            event_visible = False
        else:
            event_visible = True
        if event[1][0:4] == 'turn':  # identifying turning commands
            is_turn = True
        else:
            is_turn = False
        navigation_event_array.append(NavigationEvent(event[1], gps_location, event_visible, True, is_turn))  # (coordinates, is_drawable, is_viewable, is_turn)

    return np.array(route_polyline_vector, dtype=np.float32), navigation_event_array  # return route polyline and navigation events array


def get_relative_route(car_location, array, scale):  # returns the route relative to car's current location. Also scale to relate GPS units to mm
    point = np.array(car_location, dtype=np.float64)
    relative_route_gps = (array - point)*scale  # subtract point from every element of array and adjust scale
    return relative_route_gps


def convert_world_to_cam(point_array, rotation_matrix, translation_vector, intrinsic_parameters):
    world_coords = np.array([[point_array[0]], [point_array[1]], [point_array[2]]])
    transformed = rotation_matrix * world_coords + translation_vector
    transformed = transformed.tolist()
    u = intrinsic_parameters[0] * (transformed[0][0] / transformed[2][0]) + intrinsic_parameters[2]
    v = intrinsic_parameters[1] * (transformed[1][0] / transformed[2][0]) + intrinsic_parameters[3]
    return [int(u), int(v)]


def nothing(x):
    pass


#  route_data_process
def get_route_data(conn1, live, intrinsic_parameters, global_route, route_instructions, conn2):

    # initialize snapping position and direction --------------
    global_route_preserved = global_route  # used for snapping
    leg = 0  # efficiency measure
    compass_dampening = 0.6
    car_direction = 0
    # ---------------------------------------------------------

    f = open('R2.txt', 'r')  # open data file
    data = f.readline().split(",")  # reads first line of data

    car_location_snapped, leg, direction_snapped = snap([float(data[1]), float(data[2])], global_route_preserved, leg, 2)  # snaps just lat/long for now. Maybe implement altitude later
    car_location = [car_location_snapped[0], car_location_snapped[1], 0]
    road_width = 10
    global_route = draw_road_2(global_route, road_width, 1000000)

    scale = 10000000  # relates GPS units to millimeters. 110550039 was the calculated value, it should be correct.
    x_theta = 0
    z_theta = 0
    x_translation = 0
    y_translation = 100
    z_translation = 1
    clip_rotation_val = 0

    clock = 0
    timer = 0

    timer10 = time.time()
    timer20 = time.time()

    while True:
        start = time.time()  # only used for tracking frame rate

        # RECIEVE NEW DATA FROM QUEUE ----------------------------------------
        if not conn2.empty():
            aa = conn2.get()
            try:
                x_theta = np.deg2rad(aa[0])
                z_theta = np.deg2rad(aa[2])
                x_translation = aa[3]
                y_translation = aa[4]
                z_translation = aa[5]
                clip_rotation_val = np.deg2rad(aa[6])
                scale = aa[7]*10000
            except IndexError:
                pass


        data = f.readline().split(",")  # reads next line of data
        car_location_snapped, leg, direction_snapped = snap([np.float64(data[1]), np.float64(data[2])], global_route_preserved, leg, 2)  # snaps just lat/long for now. Maybe implement altitude later

        # CAR POSITION AND DIRECTION -----------------------------------------
        car_location = [car_location_snapped[0], car_location_snapped[1], 0]
        car_direction += (direction_snapped - car_direction)*compass_dampening  # this is written this way to dampen the rotation a bit

        # print(car_location)
        # TRANSFORMATION MATRIX ----------------------------------------------
        y_theta = np.deg2rad(car_direction)
        x_t = np.deg2rad(270)
        z_t = np.deg2rad(180)
        tilt_angle = np.deg2rad(2.7)

        x_rotation_matrix = np.matrix([[1, 0, 0], [0, np.cos(x_t), -np.sin(x_t)], [0, np.sin(x_t), np.cos(x_t)]])
        y_rotation_matrix = np.matrix([[np.cos(y_theta), 0, np.sin(y_theta)], [0, 1, 0], [-np.sin(y_theta), 0, np.cos(y_theta)]])
        z_rotation_matrix = np.matrix([[np.cos(z_t), -np.sin(z_t), 0], [np.sin(z_t), np.cos(z_t), 0], [0, 0, 1]])

        tilt_matrix = np.matrix([[np.cos(tilt_angle), 0, np.sin(tilt_angle)], [0, 1, 0], [-np.sin(tilt_angle), 0, np.cos(tilt_angle)]])

        rotation_matrix = z_rotation_matrix * y_rotation_matrix * x_rotation_matrix  # we can hard-code the resultant matrix after we're done testing. It will be more efficient at runtime.
        rotation_matrix *= tilt_matrix
        translation_vector = np.array([[x_translation], [y_translation], [z_translation]], np.int32)  # affects the height of the camera, and the 'zoom' in a sense. This will have to be calibrated to match the height of the camera in the car

        # ROUTE DATA ---------------------------------------------------------
        # Getting Relative Route Coordinates
        relative_route = get_relative_route(car_location, global_route, scale)
        clip_rotation = -y_theta + clip_rotation_val
        clip_point = np.array([0, 0, 0])  # will be origin as long as the world is moved around the car, if we change this, the clip_point will have to match the car's position
        relative_route = clip_array(relative_route, clip_point, clip_rotation, global_route, scale, True)

        # Perspective Transform on Relative Route Coordinates
        route_pt = []
        for i in relative_route:
            route_pt.append(convert_world_to_cam(i, rotation_matrix, translation_vector, intrinsic_parameters))

        # EVENT DATA ---------------------------------------------------------
        relative_event_coordinates = []
        for event in route_instructions:
            event.set_relative_event_coordinates(get_relative_route(car_location, event.event_coordinates, scale))
            relative_event_coordinates.append(event.relative_event_coordinates)  # gets relative GPS coordinates for events
        relative_event_coordinates = np.array(relative_event_coordinates)

        events_pt_is_visible = clip_array(relative_event_coordinates, clip_point, clip_rotation, global_route, scale, False)
        for i in range(0, len(events_pt_is_visible)):
            route_instructions[i].set_visibility(events_pt_is_visible[i])

        # Perspective Transform on Relative Event Coordinates
        for event in route_instructions:
            event.set_event_screen_coordinates(convert_world_to_cam(event.relative_event_coordinates, rotation_matrix, translation_vector, intrinsic_parameters))

        # Get Distance Between Each Event Object and the Vehicle
        for event in route_instructions:
            event.get_dist(car_location)

        # Delete line below ... this is an artifact from testing the 3D engine
        car_location_pt = convert_world_to_cam(clip_point, rotation_matrix, translation_vector, intrinsic_parameters)  # Used to draw the car's location as a red dot. Note, this is just at the origin and this line can be deleted. It was just used for testing.
        # Delete line above ...

        # SEND DATA THROUGH QUEUE --------------------------------------------
        conn1.put([clock, car_location_pt, route_pt, route_instructions])  # send data to queue for the main process to grab. For live scenario, we don't need a timestamp.

        # MISC ---------------------------------------------------------------
        clock += 0.1  # replace this with actual timestamp data from text file

        delay = time.time()  # the delay experienced by running all of the matrix operations etc, for each 10 Hz frame. If this delay exceeds 100 ms, then the data will become out of sync. This will not likely happen since the operations are (relatively) very fast.
        # print(delay-start)
        if 0.1 - (delay - start) > 0:  # since the code takes a certain amount of time to execute, this adds an addition delay to make the total loop delay equal to 100 ms
            time.sleep(0.1 - (delay - start))
        else:
            print("Loop calculations exceeded desired loop delay!")  # this is bad, but won't likely happen
            pass


#  main_process
if __name__ == '__main__':
    cv2.namedWindow('Main1')

    cv2.createTrackbar('x_rotation (roll)', 'Main1', 270, 360, nothing)  # old value: 270
    cv2.createTrackbar('y_rotation (yaw)', 'Main1', 0, 360, nothing)
    cv2.createTrackbar('z_rotation (pitch)', 'Main1', 180, 360, nothing)  # old value: 180

    cv2.createTrackbar('x_translation', 'Main1', 0, 500, nothing)
    cv2.createTrackbar('y_translation', 'Main1', 400, 20000, nothing)
    cv2.createTrackbar('z_translation', 'Main1', 1, 2000, nothing)
    cv2.createTrackbar('clip_rotation', 'Main1', 270, 360, nothing)
    cv2.createTrackbar('scale1', 'Main1', 5000, 50000, nothing)

    live = False  # indicates whether program is operating with live or recorded data

    if not live:
        filename = "R2.mov"
        video_dimensions = [640, 360]  # width/height
        # camera_parameters = [2797.43, 2797.43, video_dimensions[0]/2, video_dimensions[1]/2]  # fx, fy, cx, cy (480p iPhone 5s video)
        camera_parameters = [1229, 1153, video_dimensions[0] / 2, video_dimensions[1] / 2]  # fx, fy, cx, cy (480p iPhone 6 video)
        cap = cv2.VideoCapture(filename)
    else:
        video_dimensions = [640, 360]  # width/height
        # camera_parameters = [2797.43, 2797.43, video_dimensions[0]/2, video_dimensions[1]/2]  # fx, fy, cx, cy (480p iPhone 5s video)
        camera_parameters = [1229, 1153, video_dimensions[0]/2, video_dimensions[1]/2]  # fx, fy, cx, cy (480p iPhone 6 video)
        cap = cv2.VideoCapture(0)  # live video still needs to be properly implemented

    road_alpha = 0.4
    ui_alpha = 0.8

    api_key = 'AIzaSyDHOw34O0k8qDJ-td0jJhmi7GskJVffY64'
    origin = [43.266967, -79.959068]  # replace this with car's gps coordinates. Make an initialization function which samples the car's current location, then queries Google for the global route
    destination = [43.259140, -79.941978]

    global_route, route_events = fetch_route_google_api(api_key, origin, destination, False)  # Numpy array of polyline data. Boolean arg for altitudes

    q1 = Queue()  # send data from child to parent
    q2 = Queue()  # send data from parent to child
    p = Process(target=get_route_data, args=(q1, live, camera_parameters, global_route, route_events, q2,))
    p.start()

    start = time.time()

    ready = False
    while not ready:  # waits for the other process to begin sending data before beginning the main process loop
        if not q1.empty():
            route_data = q1.get()
            try:
                time_stamp = route_data[0]  # we might have to make minor modifications to this for live video
                car_location_image = route_data[1]  # data is received in image coordinates
                route_points_image = route_data[2]  # data is received in image coordinates
                route_instructions = route_data[3]
                ready = True
            except IndexError:
                pass

    counter = 0
    while True:  # main process loop

        ret, img = cap.read()

        end = time.time()
        start = time.time()

        # get data from route data process
        if not live and round(time_stamp, 1) < round(cap.get(cv2.CAP_PROP_POS_MSEC)/1000, 1) and not q1.empty():  # ensures that timestamp in recorded data matches current img in video
            route_data = q1.get()
            time_stamp = route_data[0]
            car_location_image = route_data[1]
            route_points_image = route_data[2]
            route_instructions = route_data[3]
            q2.put([cv2.getTrackbarPos('x_rotation (roll)', 'Main1'), cv2.getTrackbarPos('y_rotation (yaw)', 'Main1'), cv2.getTrackbarPos('z_rotation (pitch)', 'Main1'), cv2.getTrackbarPos('x_translation', 'Main1'), cv2.getTrackbarPos('y_translation', 'Main1'), cv2.getTrackbarPos('z_translation', 'Main1'), cv2.getTrackbarPos('clip_rotation', 'Main1'), cv2.getTrackbarPos('scale1', 'Main1')])

        elif live:
            while not q1.empty():  # gets most recently added element in the queue. The queue won't likely fill up anyway since queue is checked at ~30 Hz and data is added to the queue at 10 Hz. This is just to make the program handle lags well.
                route_data = q1.get()  # if data is live, then take new data at the queue immediately as it arrives
                # time_stamp = route_data[0]  # timestamps don't matter for live serial data... this is used just for testing
                car_location_image = route_data[1]
                route_points_image = route_data[2]
                # event_list = route_data[3]
                # event_list_visibility = route_data[4]
                route_instructions = route_data[3]

        road_overlay = img.copy()
        ui_overlay = img.copy()
        output = img.copy()

        # DESIRED UI - INEFFICIENT, HOWEVER :(
        road_gradient = img.copy()
        cv2.fillPoly(road_gradient, [np.array(route_points_image, np.int32)], (255, 255, 75))
        road_gradient = gradient_blend(img, road_gradient, 0.5)
        # cv2.polylines(road_gradient, [np.array(route_points_image, np.int32)], True, (255, 255, 75), 0, cv2.LINE_AA)
        # output = gradient_blend(img, road_gradient, 0.7)
        # --------------------------------------------------------------------------------------------------------------

        # road_gradient = img.copy()
        # cv2.fillPoly(road_gradient, [np.array(route_points_image, np.int32)], (255, 255, 75))
        # cv2.polylines(road_gradient, [np.array(route_points_image, np.int32)], True, (255, 255, 255), 0, cv2.LINE_AA)
        # output = gradient_blend(img, road_gradient, 0.3)

        # for event in route_instructions:
        #     # print(event.distance_from_vehicle)
        #     if event.is_viewable and event.distance_from_vehicle < 0.001:
        #         point_size = round(1/event.distance_from_vehicle/1000, 1)
        #         w = 50
        #         h = 40
        #         line = 150
        #         cv2.line(ui_overlay, event.event_screen_coordinates, (event.event_screen_coordinates[0], event.event_screen_coordinates[1]-line), (255, 255, 75), 1, cv2.LINE_AA)
        #         # cv2.line(overlay, event.event_screen_coordinates, (event.event_screen_coordinates[0], event.event_screen_coordinates[1]-line), (255, 255, 75), 1)
        #         cv2.circle(output, event.event_screen_coordinates, int(point_size), (255, 255, 75), -1)
        #         cv2.rectangle(ui_overlay, (event.event_screen_coordinates[0]-w, event.event_screen_coordinates[1]-(line+h)), (event.event_screen_coordinates[0]+w, event.event_screen_coordinates[1]-line), (255, 255, 75), 1, cv2.LINE_AA)
        #         cv2.rectangle(road_overlay, (event.event_screen_coordinates[0]-w, event.event_screen_coordinates[1]-(line+h)), (event.event_screen_coordinates[0]+w, event.event_screen_coordinates[1]-line), (255, 255, 75), -1)
        #         cv2.putText(ui_overlay, 'Hello World!', (event.event_screen_coordinates[0], event.event_screen_coordinates[1]-(line + int(h/2))), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,75), 2, cv2.LINE_AA)
        #         # output[30:30 + 100, 30:30 + 100] = blurred[30:30 + 100, 30:30 + 100]

        # cv2.addWeighted(img, road_alpha, output, 1 - road_alpha, 0, output)
        # cv2.addWeighted(ui_overlay, ui_alpha, output, 1 - ui_alpha, 0, output)
        # cv2.imshow('Main', output)
        # cv2.imshow('Main', mask)
        cv2.imshow('Main', road_gradient)
        if cv2.waitKey(30) == ord('q'):
            break

    cv2.destroyAllWindows()
