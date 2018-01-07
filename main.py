from multiprocessing import Process, Queue
import cv2
import numpy as np
import requests
import polyline
import json
import time


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


def get_altitude(key, lat, long):
    api_url = 'https://maps.googleapis.com/maps/api/elevation/json?locations={0},{1}&key{2}'.format(str(lat), str(long), key)
    output = requests.get(api_url)
    altitude = json.loads(output.text)["results"][0]["elevation"]
    return round(altitude, 4)


def fetch_route_google_api(key, start, end, fetch_altitudes):
    api_url = 'https://maps.googleapis.com/maps/api/directions/json?origin={0},{1}&destination={2},{3}&key{4}'.format(str(start[0]), str(start[1]), str(end[0]), str(end[1]), key)
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

    return np.array(route_polyline_vector, dtype=np.float32)  # return numpy array


def get_relative_route(car_location, array, scale):  # returns the route relative to car's current location. Also scale to relate GPS units to mm
    point = np.array(car_location, dtype=np.float32)
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
def get_route_data(conn, live, intrinsic_parameters):
    api_key = 'AIzaSyDHOw34O0k8qDJ-td0jJhmi7GskJVffY64'
    origin = [43.256963, -79.925822]  # replace this with car's gps coordinates. Make an initialization function which samples the car's current location, then queries Google for the global route
    destination = [43.259598, -79.923227]

    # global_route = fetch_route_google_api(api_key, origin, destination, False)  # Numpy array of polyline data. Boolean arg for altitudes
    global_route = [[-79.92581, 43.25696, 0], [-79.92589, 43.25756, 0], [-79.92562, 43.25757, 0],
                    [-79.9238, 43.25762, 0], [-79.92288, 43.25767, 0], [-79.92281, 43.25764, 0],
                    [-79.92177, 43.25766, 0], [-79.92098, 43.25769, 0], [-79.91845, 43.25773, 0],
                    [-79.91574, 43.2578, 0], [-79.91307, 43.25787, 0], [-79.91211, 43.25791, 0],
                    [-79.91161, 43.25792, 0], [-79.91153, 43.25794, 0], [-79.9114, 43.25801, 0],
                    [-79.91111, 43.25804, 0], [-79.9104, 43.25812, 0], [-79.90693, 43.2585, 0], [-79.9055, 43.25869, 0],
                    [-79.90441, 43.25884, 0], [-79.90327, 43.25901, 0], [-79.90239, 43.25913, 0],
                    [-79.90237, 43.2593, 0], [-79.90225, 43.25965, 0], [-79.90189, 43.26039, 0],
                    [-79.90074, 43.26301, 0], [-79.90106, 43.26309, 0]]

    # make a function which parses the Google API response, then organizes the event data in this format:
    # also maybe make a separate function for the event array which doesn't draw the ones that aren't on screen

    event_coordinates = [[-79.92589, 43.25756, 0],[-79.92281, 43.25764, 0], [-79.91574, 43.2578, 0], [-79.90106, 43.26309, 0]]
    event_commands = ["Event 1", "Event 2", "Event 3", "Event 4"]

    road_width = 0.000015
    global_route = draw_road(np.array(global_route), road_width)  # global route remains unaltered for the remainder of the code.
    scale = 10000000  # relates GPS units to millimeters. 110550039 was the calculated value, it should be correct.

    clock = 0

    f = open('serial_output_2.txt', 'r')  # open data file

    while True:
        start = time.time()  # only used for tracking frame rate

        # get data from file
        data = f.readline().split(",")  # reads next line of data
        car_location = [float(data[1]), float(data[2]), 0]
        car_direction = float(data[3])

        # generate relative route based on car position and direction
        relative_route = get_relative_route(car_location, global_route, scale)  # !!! this could be more efficient. Right now we are moving the world camera coordinates around the camera. We should move the camera around the world. Although, now that this is a separate process, efficiency isn't too big of a deal. Re-program this if you have time.
        relative_event_coordinates = get_relative_route(car_location, event_coordinates, scale)

        # Compute rotation matrix ----------------------------------------------------------------------
        # If we end up using altitudes, modify the micro-controller code to get car pitch from accelerometer, then modify x_theta and/or z_theta accordingly
        x_theta = 1571 / 1000.0 - 3.14
        z_theta = 6280 / 1000.0 - 3.14
        y_theta = np.deg2rad(car_direction)  # add an offset to this, I think that the world currently thinks South is East (?). ie. Add Pi/2 to the recorded data

        x_rotation_matrix = np.matrix(
            [[1, 0, 0], [0, np.cos(x_theta), -np.sin(x_theta)], [0, np.sin(x_theta), np.cos(x_theta)]])
        y_rotation_matrix = np.matrix(
            [[np.cos(y_theta), 0, np.sin(y_theta)], [0, 1, 0], [-np.sin(y_theta), 0, np.cos(y_theta)]])
        z_rotation_matrix = np.matrix(
            [[np.cos(z_theta), -np.sin(z_theta), 0], [np.sin(z_theta), np.cos(z_theta), 0], [0, 0, 1]])

        rotation_matrix = z_rotation_matrix * y_rotation_matrix * x_rotation_matrix  # we can hard-code the resultant matrix after we're done testing. It will be more efficient at runtime.
        translation_vector = np.array([[0], [100], [1]], np.int32)  # affects the height of the camera, and the 'zoom' in a sense. This will have to be calibrated to match the height of the camera in the car

        # Clipping & Perspective Transform -------------------------------------------------------------
        clip_rotation = -4712 / 1000.0 - np.pi / 2
        clip_point = np.array([0, 0, 0])  # will be origin as long as the world is moved around the car, if we change this, the clip_point will have to match the car's position
        relative_route = clip_array(relative_route, clip_point, clip_rotation, global_route, scale, True)

        route_pt = []  # "route perspective transform"
        for i in relative_route:
            route_pt.append(convert_world_to_cam(i, rotation_matrix, translation_vector, intrinsic_parameters))

        events_pt_is_visible = clip_array(relative_event_coordinates, clip_point, clip_rotation, global_route, scale, False)
        events_pt = []  # "events perspective transform"
        for e in relative_event_coordinates:
            events_pt.append(convert_world_to_cam(e, rotation_matrix, translation_vector, intrinsic_parameters))

        car_location_pt = convert_world_to_cam(clip_point, rotation_matrix, translation_vector, intrinsic_parameters)  # Used to draw the car's location as a red dot. Note, this is just at the origin and this line can be deleted. It was just used for testing.

        conn.put([clock, car_location_pt, route_pt, events_pt, events_pt_is_visible])  # send data to queue for the main process to grab. For live scenario, we don't need a timestamp.

        clock += 0.1  # replace this with actual timestamp data from text file

        delay = time.time()  # the delay experienced by running all of the matrix operations etc, for each 10 Hz frame. If this delay exceeds 100 ms, then the data will become out of sync. This will not likely happen since the operations are (relatively) very fast.

        if 0.1 - (delay - start) > 0:  # since the code takes a certain amount of time to execute, this adds an addition delay to make the total loop delay equal to 100 ms
            time.sleep(0.1 - (delay - start))
        else:
            print("Loop calculations exceeded desired loop delay!")  # this is bad, but won't likely happen
            pass

        end = time.time()
        # print("Route Data Process, Loop: " + str(round(1/(end - start), 1)) + " Hz")


#  main_process
if __name__ == '__main__':

    live = False  # indicates whether program is operating with live or recorded data

    if not live:
        filename = "sample_video480.mov"
        video_dimensions = [640, 360]  # width/height
        camera_parameters = [2797.43, 2797.43, video_dimensions[0]/2, video_dimensions[1]/2]  # fx, fy, cx, cy (480p iPhone 5s video)
        cap = cv2.VideoCapture(filename)
    else:
        video_dimensions = [640, 360]  # width/height
        camera_parameters = [2797.43, 2797.43, video_dimensions[0]/2, video_dimensions[1]/2]  # fx, fy, cx, cy (480p iPhone 5s video)
        cap = cv2.VideoCapture(0)  # live video still needs to be properly implemented

    window_scale = 1
    cv2.namedWindow('Main', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Main', int(video_dimensions[0]*window_scale), int(video_dimensions[1]*window_scale))

    # create new process for route data
    q = Queue()
    p = Process(target=get_route_data, args=(q, live, camera_parameters,))
    p.start()

    start = time.time()

    ready = False
    while not ready:  # waits for the other process to begin sending data before beginning the main process loop
        if not q.empty():
            route_data = q.get()
            time_stamp = route_data[0]  # we might have to make minor modifications to this for live video
            car_location_image = route_data[1]  # data is received in image coordinates
            route_points_image = route_data[2]  # data is received in image coordinates
            event_list = route_data[3]
            event_list_visibility = route_data[4]
            ready = True

    while cap.isOpened():  # main process loop
        ret, frame = cap.read()
        end = time.time()
        # print("Main Process, Loop: ~" + str(round(1/(end - start), 1)) + " FPS")  # instantaneous calculated frequency, actual frame-rate is probably closer to 30 FPS
        start = time.time()

        # get data from route data process
        if not live and round(time_stamp, 1) < round(cap.get(cv2.CAP_PROP_POS_MSEC)/1000, 1) and not q.empty():  # ensures that timestamp in recorded data matches current frame in video
            route_data = q.get()
            time_stamp = route_data[0]
            car_location_image = route_data[1]
            route_points_image = route_data[2]
            event_list = route_data[3]
            event_list_visibility = route_data[4]
        elif live:
            while not q.empty():  # gets most recently added element in the queue. The queue won't likely fill up anyway since queue is checked at ~30 Hz and data is added to the queue at 10 Hz. This is just to make the program handle lags well.
                route_data = q.get()  # if data is live, then take new data at the queue immediately as it arrives
                # time_stamp = route_data[0]  # timestamps don't matter for live serial data... this is used just for testing
                car_location_image = route_data[1]
                route_points_image = route_data[2]
                event_list = route_data[3]
                event_list_visibility = route_data[4]

        # print("Video Timestamp:" + str(round(cap.get(cv2.CAP_PROP_POS_MSEC)/1000, 1)))
        # print("Data Timestamp:" + str(round(time_stamp, 1)))
        # print("--")

        cv2.polylines(frame, [np.array(route_points_image, np.int32)], True, (255, 255, 255), 1)
        cv2.circle(frame, (int(car_location_image[0]), int(car_location_image[1])), 35, (0, 0, 255), -1)

        for i in range(0, len(event_list)-1):
            if event_list_visibility[i]:
                cv2.circle(frame, (int(event_list[i][0]), int(event_list[i][1])), 5, (0, 255, 255), -1)

        cv2.imshow('Main', frame)

        cv_key = cv2.waitKey(30) & 0xFF == ord('q')

    cap.release()
    cv2.destroyAllWindows()