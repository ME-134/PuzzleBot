#!/usr/bin/env python3

from ast import arg
import enum

import rospy

import cv2
from vision import cvt_color

from sensor_msgs.msg   import Image

import numpy as np

from thomas_detector import ThomasDetector, ThomasPuzzlePiece, ToThomasPuzzlePiece
from puzzle_grid import PuzzleGrid
import vision

class SolverTask(enum.Enum):
    # HIGH-LEVEL TASKS
    SeparatePieces = 0
    PutPiecesTogether = 1
    SeparateOverlappingPieces = 2

    # LOW-LEVEL TASKS
    GetView = 3
    GetViewCleared = 4
    GetViewPuzzle = 5
    MoveArm = 6
    LiftPiece = 7
    PlacePiece = 8
    MatePiece = 10

    # MISC TASKS
    InitAruco = 9

class TaskStack:
    def __init__(self):
        self.tasks = list()
        self.taskdata = list()
    def push(self, task, task_data=None):
        if task_data is None:
            task_data = dict()
        self.tasks.append(task)
        self.taskdata.append(task_data)
    def append(self, *args, **kwargs):
        return self.push(*args, **kwargs)
    def pop(self):
        task = self.tasks.pop()
        taskdata = self.taskdata.pop()
        return task, taskdata
    def peek(self):
        return self.tasks[-1], self.taskdata[-1]
    def __len__(self):
        return len(self.tasks)
    def __bool__(self):
        return len(self.tasks) > 0
    def __repr__(self):
        s = "Task Stack: \n"
        s += "[Top]\n"
        for i, (task, taskdata) in enumerate(reversed(list(zip(self.tasks, self.taskdata)))):
            s += f"{i+1}. " + str(task)
            if taskdata:
                s += " | Task Data: " + str(taskdata)
            s += "\n"
        s += "[Bottom]"
        return s

class Status(enum.Enum):
    Ok = 0
    PieceDropped = 1

    def ok(self):
        return self == Status.Ok

    def assert_ok(self):
        if not self.ok():
            errmsg = f"[Solver] Unknown error: {self}"
            rospy.logerror(errmsg)
            raise RuntimeError(errmsg)

class Solver:
    def __init__(self, detector):

        self.pub_clearing_plan = rospy.Publisher("/solver/clearing_plan",  Image, queue_size=1)
        self.pub_contour_matching = rospy.Publisher("/solver/matching",  Image, queue_size=1)
        self.vision = vision.VisionMatcher('/home/me134/me134ws/src/HW1/done_exploded_colored7.jpg')

        # Stack
        self.tasks = TaskStack()
        self.tasks.push(SolverTask.PutPiecesTogether)
        self.tasks.push(SolverTask.SeparatePieces)
        self.tasks.push(SolverTask.InitAruco)
        self.tasks.push(SolverTask.GetView)

        # Series of actions to perform
        self.action_queue = []

        self.puzzle_grid = PuzzleGrid()
        self.num_pieces = 20
        self.pieces_cleared = 0
        self.separated_loc = np.array([220, 100])
        self.separated_spacing = 100
        self.pieces_solved = 0

        # Snapshot of the list of pieces seen by the detector.
        # Only update when robot arm is not in camera frame.
        # Use GetView task to achieve this.
        self.piece_list = list()

        # Same detector as the one used by the controller
        self.detector = detector

    # Public methods
    def notify_action_completed(self, status):
        '''
        Tell the Solver that the current task has been completed.
        The solver then internally figures out what to do next.

        Args: status, which is either OK or some sort of error.
        Return: None
        '''
        print("notify action complete: ", self.tasks)
        if not self.tasks:
            rospy.logwarn("[Solver] Action completed recieved but there are no current tasks!")
            return

        # Default to the current task being completed
        # We can add it back to the task stack if needed (esp. for high-level tasks)
        curr_task, task_data = self.tasks.pop()

        # HIGH-LEVEL TASKS
        if curr_task == SolverTask.SeparatePieces:
            status.assert_ok()

            self.pieces_cleared += 1
            if self.pieces_cleared == self.num_pieces:
                rospy.loginfo(f"[Solver] Cleared all {self.num_pieces} pieces, moving on to next task.")
                return

            # To decide whether we are done separating pieces, we need a clear view of the board first.
            self.tasks.append(SolverTask.SeparatePieces)
            self.tasks.append(SolverTask.GetView)

        elif curr_task == SolverTask.PutPiecesTogether:
            status.assert_ok()

            self.pieces_solved += 1
            if self.pieces_solved == self.num_pieces:
                rospy.loginfo(f"[Solver] Solved all {self.num_pieces} pieces! Moving on to next task.")
                return

            # Get a view of the available pieces
            self.tasks.append(SolverTask.PutPiecesTogether)
            self.tasks.append(SolverTask.GetViewCleared, task_data={'merge': False})

        elif curr_task == SolverTask.SeparateOverlappingPieces:
            # Out of scope for now
            raise NotImplementedError()

        # LOW-LEVEL TASKS
        elif curr_task == SolverTask.GetView:
            status.assert_ok()

            # The camera should have a clear view of the pieces now.
            self.detector.snap()
            self.piece_list = self.detector.pieces.copy()

        elif curr_task == SolverTask.GetViewCleared:
            status.assert_ok()

            # The camera should have a clear view of the border-region pieces now.
            self.detector.snap(black_list = [self.get_puzzle_region()], merge=task_data['merge'])
            self.piece_list = self.detector.pieces.copy()

        elif curr_task == SolverTask.GetViewPuzzle:
            status.assert_ok()

            # The camera should have a clear view of the puzzle region now.
            self.detector.snap(white_list = [self.get_puzzle_region()], merge=task_data['merge'])
            self.piece_list = self.detector.pieces.copy()

        # No action required assuming status is OK.
        elif curr_task == SolverTask.MoveArm:
            status.assert_ok()
        elif curr_task == SolverTask.LiftPiece:
            status.assert_ok()
        elif curr_task == SolverTask.PlacePiece:
            status.assert_ok()
        elif curr_task == SolverTask.MatePiece:
            status.assert_ok()

        # MISC TASKS
        elif curr_task == SolverTask.InitAruco:
            status.assert_ok()
        else:
            raise RuntimeError(f"Unknown task: {curr_task}")

    def apply_next_action(self, controller):
        '''
        This function will continuously apply the action at the top of the stack.
        To break out of the recursive loop, simply "return".
        '''
        print("apply next action: ", self.tasks)

        if not self.tasks:
            rospy.loginfo("[Solver] No current tasks, sending Idle command.")
            controller.idle()
            return

        curr_task, task_data = self.tasks.peek()
        rospy.loginfo(f"[Solver] Current task is {curr_task}.")

        if curr_task == SolverTask.GetView:
            # If we want to get the view, simply go to the reset position, where
            # the robot arm is not in the camera frame.
            rospy.loginfo("[Solver] Sending Reset command.")
            controller.reset()
            return

        elif curr_task == SolverTask.GetViewCleared:
            # Get a clear view of all the cleared puzzle pieces

            # If the arm is in the puzzle region, we already have a view of the cleared pieces
            x, y = self.get_arm_location(controller)
            if self.point_in_puzzle_region(x, y, buffer=-20):
                rospy.loginfo("[Solver] Arm is in the puzzle region, moving on to the next task.")

                # End this task and attempt to do the next one.
                self.tasks.pop()

            else:
                rospy.loginfo("[Solver] Arm is not in the puzzle region, resetting robot.")
                controller.reset()
                return

        elif curr_task == SolverTask.GetViewPuzzle:
            # Get a clear view of the puzzle region

            # If the arm is not in the puzzle region, we already have a view of the puzzle region
            x, y = self.get_arm_location(controller)
            if not self.point_in_puzzle_region(x, y, buffer=20):
                rospy.loginfo("[Solver] Arm is not in the puzzle region, moving on to the next task.")

                # End this task and attempt to do the next one.
                self.tasks.pop()

            else:
                rospy.loginfo("[Solver] Arm is in the puzzle region, resetting robot.")
                controller.reset()
                return

        elif curr_task == SolverTask.InitAruco:
            self.detector.init_aruco()
            self.tasks.pop()

        elif curr_task == SolverTask.MoveArm:
            controller.move_to_pixelcoords(task_data['dest'], task_data.get('turn', 0))
            return

        elif curr_task == SolverTask.LiftPiece:
            controller.lift_piece()
            return

        elif curr_task == SolverTask.PlacePiece:
            controller.place_piece(jiggle=task_data['jiggle'])
            return

        elif curr_task == SolverTask.MatePiece:
            # Matches the two biggest pieces
            # Makes first piece the main puzzle
            self.tasks.pop()
            pieces = sorted(self.detector.pieces, key=lambda x: -x.area)
            loc1, loc2 = np.array(pieces[0].get_location()), np.array(pieces[1].get_location())
            delta = loc1 - loc2
            delta = delta / np.linalg.norm(delta)
            if np.arccos(np.dot(delta, [-1/np.sqrt(2), -1/np.sqrt(2)])) > 1.4:
                pieces[0], pieces[1] = pieces[1], pieces[0]
            self.puzzle_grid.piece = (pieces[0])
            dx, dy, dtheta, sides = ToThomasPuzzlePiece(pieces[1]).find_contour_match(ToThomasPuzzlePiece(pieces[0]), match_threshold=7, return_sides=True)
            if dx == 0 and dy == 0:
                piece_destination = self.find_available_piece_spot(pieces[1], 0)
                self.tasks.push(SolverTask.PlacePiece, task_data={'jiggle': False})
                self.tasks.push(SolverTask.MoveArm, task_data={'dest': piece_destination, 'turn': 0})
            else:
                self.tasks.push(SolverTask.PlacePiece, task_data={'jiggle': True})
                self.tasks.push(SolverTask.MoveArm, task_data={'dest': pieces[1].get_location() + np.array([dx, dy]), 'turn': dtheta})
            self.tasks.push(SolverTask.LiftPiece)
            self.tasks.push(SolverTask.MoveArm, task_data={'dest': pieces[1].get_location()})
            plan_img = np.zeros((1080, 1720, 3), dtype=np.uint8) + 255
            for side in sides:
                plan_img[side] = [255, 255, 0]
            dummy_piece = pieces[1].copy()
            dummy_piece.rotate(dtheta)
            dummy_piece.move_to(np.array(dummy_piece.get_location()[0]) + dx, np.array(dummy_piece.get_location()[1]) + dy)
            plan_img[pieces[0].mask.astype(bool)] += np.array(pieces[0].color).astype(np.uint8)
            plan_img[dummy_piece.mask.astype(bool)] += np.array(pieces[1].color).astype(np.uint8)
            self.pub_contour_matching.publish(self.detector.bridge.cv2_to_imgmsg(plan_img, "bgr8"))

        # HIGH-LEVEL TASKS
        elif curr_task == SolverTask.SeparatePieces:
            # We are trying to clear the center of the playground to make room for
            # the completed puzzle. At the same time, we are rotating all the pieces
            # to their correct orientation.

            for piece in sorted(self.piece_list, key=lambda piece: piece.xmin):

                # rotation_offset = self.get_rotation_offset(piece.img)
                rotation_offset = piece.get_aligning_rotation()
                ### TEMP
                rotation_offset *= -1

                rotation_offset += np.pi / 4
                rotation_offset %= np.pi / 2
                rotation_offset -= np.pi / 4

                rospy.loginfo(f"rotation offset: {rotation_offset}")
                ### END TEMP

                # Select piece which is not rotated correctly or is in the puzzle region
                threshold_rotation_error = 0.13
                if abs(rotation_offset) > threshold_rotation_error:
                    break
                if piece.overlaps_with_region(self.get_puzzle_region()):
                    break
            else:
                # All pieces have been cleared from the puzzle region
                rospy.loginfo("[Solver] No pieces left to separate, continuing to next solver task.")

                # TODO: in the future, we might want to not pop, so that
                # we can recover from errors.
                self.tasks.pop()
                return self.apply_next_action(controller)

            piece_origin = piece.get_center()
            piece_destination = self.find_available_piece_spot(piece, rotation_offset)

            # Add in reverse
            self.tasks.push(SolverTask.GetView)
            self.tasks.push(SolverTask.PlacePiece, task_data={'jiggle': False})
            self.tasks.push(SolverTask.MoveArm, task_data={'dest': piece_destination, 'turn': rotation_offset})
            self.tasks.push(SolverTask.LiftPiece)
            self.tasks.push(SolverTask.MoveArm, task_data={'dest': piece_origin})

            # Continue onto the next action
            return self.apply_next_action(controller)

        elif curr_task == SolverTask.PutPiecesTogether:
            # All pieces should now be on the border and oriented orthogonal.
            # Select pieces from the border and put them in the right place
            radial = True

            locations, rots, scores = self.vision.match_all(self.piece_list)
            done = False
            hackystack = TaskStack() # temporary hacky solution
            temp_grid = PuzzleGrid()

            plan_img = self.detector.latestImage.copy()

            # Mark out puzzle territory in green
            plan_img[self.get_puzzle_region_as_slice()] += np.array([0, 40, 0], dtype=np.uint8)

            def place_offset(location, offset=50):
                print(temp_grid.get_neighbors(location), location)
                neighbors = temp_grid.get_neighbors(location) - location
                return -neighbors.sum(axis=0) * offset

            def processpiece(i, first=False):
                offset = place_offset(locations[i]) if not first else 0
                # offset = 0
                loc = np.array([720, 380]) + temp_grid.grid_to_pixel(locations[i])
                piece = self.piece_list[i]
                # Color selected piece
                color = np.array(piece.color).astype(np.uint8)
                # plan_img[piece.bounds_slice()] += piece.color.astype(np.uint8)
                dummy_piece = piece.copy()
                dummy_piece.rotate(rots[i]*np.pi/2)
                dummy_piece.move_to(loc[0], loc[1])
                plan_img[piece.bounds_slice()] += color
                plan_img[dummy_piece.mask.astype(bool)] += color

                # NOT pushed in reverse because hackystack gets added to self.tasks in reverse
                hackystack.push(SolverTask.MoveArm, task_data={'dest': self.piece_list[i].get_location()})
                hackystack.push(SolverTask.LiftPiece)
                hackystack.push(SolverTask.MoveArm, task_data={'dest': loc + offset, 'turn': rots[i]*np.pi/2})
                hackystack.push(SolverTask.PlacePiece, task_data={'jiggle': False})
                if not first:
                    hackystack.push(SolverTask.GetViewPuzzle, task_data={'merge': False})
                    hackystack.push(SolverTask.MatePiece)

                # self.action_queue.append(call_me(self.piece_list[i].get_location(), loc, turn = rots[i]*np.pi/2, jiggle=False))
                temp_grid.occupied[tuple(locations[i])] = 1

            if radial:
                # Find top right
                target_loc = [0, 0]
                for i in range(len(self.piece_list)):
                    if np.all(locations[i] == target_loc):
                        break
                processpiece(i, first=True)
                loc_list = list()
                for i in range(len(self.piece_list)):
                    loc_list.append([i, locations[i]])
                list.sort(loc_list, key=lambda x: np.linalg.norm(x[1]-target_loc))
                while not done:
                    done = True
                    for i in range(len(self.piece_list)):
                        # Puts pieces in an order where they mate
                        if (scores[i] < 99999 and \
                            temp_grid.occupied[tuple(locations[i])] == 0 and \
                            temp_grid.does_mate(locations[i])):
                            processpiece(i)
                            done = False
            else:
                processpiece(0, first=True)
                while not done:
                    done = True
                    for i in range(1, len(self.piece_list)):
                        # Puts pieces in an order where they mate
                        if (scores[i] < 99999 and \
                            temp_grid.occupied[tuple(locations[i])] == 0 and \
                            temp_grid.does_mate(locations[i])):
                            processpiece(i)
                            done = False

            while len(hackystack) > 0:
                task, task_data = hackystack.pop()
                self.tasks.push(task, task_data=task_data)

            self.pub_clearing_plan.publish(self.detector.bridge.cv2_to_imgmsg(plan_img, "bgr8"))


            # target_piece = self.reference_pieces[self.pieces_solved]
            # for piece in self.piece_list:
            #     if piece.fully_contained_in_region(self.get_puzzle_region()):
            #         continue
            # #     if self.pieces_match(piece.img, target_piece.img):
            # #         break
            # # else:
            # #     # Error, no piece found!
            # #     rospy.logwarn(f"[Solver] Target piece #{self.peices_solved} not found! Getting a new view.")

            # #     # Attempt to recover by getting another view of the camera
            # #     self.tasks.append(SolverTask.GetView)
            # #     return self.apply_next_action(controller)

            #     piece_origin = piece.get_center()

            #     # FIXME, this isn't quite right but is a good start
            #     # piece_destination = target_piece.get_center()
            #     val = cvt_color(piece.natural_img) #* (piece.thomas_mask.reshape(piece.thomas_mask.shape[0], piece.thomas_mask.shape[1], 1) > 128)
            #     coords, rot = self.vision.calculate_xyrot(val)
            #     rospy.loginfo(f"Piece Location: {coords}, Rotation: {rot}")
            #     piece_destination = np.array(self.get_puzzle_region[[0, 1]]) + self.puzzle_grid.get_pixel_from_grid(coords)
            #     if self.puzzle_grid.occupied.sum() > 0:
            #         weight_destination = self.puzzle_grid.get_neighbor(coords)
            #         weight_pos_offset = np.array(weight_destination) - np.array(coords)
            #         weight_destination = self.puzzle_grid.get_pixel_from_grid(weight_destination)
            #         controller.move_weight(weight_destination)
            #         self.action_queue.append(lambda: controller.move_piece(piece_origin, piece_destination, turn = -rot * np.pi/2, jiggle=True))
            #     else:
            #         controller.move_piece(piece_origin, piece_destination, turn = -rot * np.pi/2, jiggle=False)
            #     return
        elif curr_task == SolverTask.SeparateOverlappingPieces:
            # Out of scope for now.
            raise NotImplementedError()

        else:
            raise RuntimeError(f"Unknown task: {curr_task}")

        # Recursive step important for correct control flow.
        self.apply_next_action(controller)

    # Private methods
    def get_arm_location(self, controller):
        # Returns arm location in pixel space
        pos = controller.get_current_position().flatten()
        x, y = pos[0], pos[1]
        return self.detector.world_to_screen(x, y)

    def point_in_puzzle_region(self, x, y, buffer=0):
        # Returns whether the screen x,y is withing the puzzle region
        xmin, ymin, xmax, ymax = self.get_puzzle_region()
        return  (xmin - buffer <= x <= xmax + buffer) \
            and (ymin - buffer <= y <= ymax + buffer)

    def get_rotation_offset(self, piece_img):
        '''
        Given an image of a piece, return how many radians clockwise it should be
        turned to be in the correct orientation.
        '''
        detector = ThomasDetector()
        detector.process(piece_img)

        if (len(detector.pieces) > 0):
            return detector.pieces[0].get_rotation_to_align(compute_bounding_box = True)

        return 0.

    def pieces_match(self, piece1_img, piece2_img):
        '''
        Given 2 pictures of pieces, return whether they are the same piece.
        For this method we can assume that the pieces are oriented the same way.
        '''
        # TODO
        return True

    def get_puzzle_region(self):
        '''
        Return (xmin, ymin, xmax, ymax) in pixel space
        representing the region where we want the solved puzzle to end up.
        '''
        # TODO
        return (600, 350, 1680, 980)

    def get_puzzle_keepouts(self):
        '''
        Return (xmin, ymin, xmax, ymax) in pixel space
        representing the region where we want the solved puzzle to end up.
        '''

        return 0

    def get_screen_region(self):
        return (0, 0, 1780, 1080)

    def get_puzzle_region_as_slice(self):
        xmin, ymin, xmax, ymax = self.get_puzzle_region()
        return (slice(ymin, ymax), slice(xmin, xmax))

    def get_aruco_regions(self):
        xmax = np.max(self.detector.aruco_corners_pixel[:, :, 0], axis=1).reshape(4, 1)
        ymax = np.max(self.detector.aruco_corners_pixel[:, :, 1], axis=1).reshape(4, 1)
        xmin = np.min(self.detector.aruco_corners_pixel[:, :, 0], axis=1).reshape(4, 1)
        ymin = np.min(self.detector.aruco_corners_pixel[:, :, 1], axis=1).reshape(4, 1)
        return np.hstack((xmin, ymin, xmax, ymax)).astype(int)

    def find_available_piece_spot(self, piece, rotation):
        '''
        Return a location on the border of the playground which has enough free space
        to put a piece there.
        '''
        # This is how we want our piece to end up
        dummy_piece_copy = piece.copy()
        dummy_piece_copy.rotate(rotation)
        dummy_piece = dummy_piece_copy.copy()

        # Mask of where there are no pieces or aruco markers
        free_space = self.detector.free_space_img.copy()

        # Add existing piece to free space
        free_space[piece.mask.astype(bool)] = 255
        free_space = cv2.dilate(free_space, None, iterations=1)

        # Scan the whole area until we find a suitable free spot for our piece
        start_x, start_y = 100, 100
        for x in range(start_x, 1780-100, 10):
            for y in range(start_y, 1080-100, 10):
                dummy_piece.move_to_no_mask(x, y)

                # Publish our plans
                # plan_img = self.detector.latestImage.copy()
                #plan_img[self.get_puzzle_region_as_slice()] += np.array([0, 0, 40], dtype=np.uint8)
                # plan_img[piece.bounds_slice()] += np.array([40, 0, 0], dtype=np.uint8)
                # plan_img[dummy_piece.bounds_slice()] += np.array([0, 20, 20], dtype=np.uint8)
                # self.pub_clearing_plan.publish(self.detector.bridge.cv2_to_imgmsg(plan_img, "bgr8"))

                if not dummy_piece.fully_contained_in_region(self.get_screen_region()):
                    continue

                # Piece cannot go to the area we designate for the solved puzzle
                if dummy_piece.overlaps_with_region(self.get_puzzle_region()):
                    break # assume puzzle region is in the bottom-right

                # Piece cannot go to the aruco areas
                flag = False
                for region in self.get_aruco_regions():
                    if dummy_piece.overlaps_with_region(region):
                        flag = True
                        break 
                if flag:
                    break

                # Piece can only go to where there are no other pieces already
                elif np.all(free_space[dummy_piece.bounds_slice(padding=15)]):
                    dummy_piece = dummy_piece_copy.copy()
                    dummy_piece.move_to(x, y)

                    plan_img = self.detector.latestImage.copy()

                    # Mark out puzzle territory in red
                    plan_img[self.get_puzzle_region_as_slice()] += np.array([0, 0, 40], dtype=np.uint8)

                    # Mark out aruco territory in red
                    pad = 10
                    for (xmin, ymin, xmax, ymax) in self.get_aruco_regions():
                        plan_img[ymin-pad:ymax+pad, xmin-pad:xmax+pad] += np.array([-20, -20, 40], dtype=np.uint8)

                    # Mark out spaces that aren't free in red
                    plan_img[free_space.astype(bool) == False] = np.array([0, 0, 200], dtype=np.uint8)

                    # Color selected piece in blue
                    plan_img[piece.bounds_slice()] += np.array([40, 0, 0], dtype=np.uint8)
                    plan_img = cv2.circle(plan_img, (piece.x_center, piece.y_center), 15, (200, 30, 30), -1)

                    # Color intended placement in green
                    plan_img[dummy_piece.bounds_slice()] += np.array([0, 20, -20], dtype=np.uint8)
                    plan_img[dummy_piece.mask.astype(bool)] = np.array([30, 200, 30], dtype=np.uint8)
                    self.pub_clearing_plan.publish(self.detector.bridge.cv2_to_imgmsg(plan_img, "bgr8"))
                    return np.array([x, y])

        rospy.logwarn("[Solver] No free spaces found!")

        # Backup
        n = self.pieces_cleared
        return self.separated_loc + [self.separated_spacing*(n%5), self.separated_spacing*(n//5)]
