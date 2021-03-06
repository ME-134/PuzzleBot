#!/usr/bin/env python3

from ast import arg
import enum

import rospy

import cv2

from sensor_msgs.msg   import Image

import numpy as np

from thomas_detector import ThomasDetector, ThomasPuzzlePiece, ToThomasPuzzlePiece
from piece_outline_detector import Detector as IlyaDetector
from puzzle_grid import PuzzleGrid

COLORLIST = ((000, 000, 255),           # Red
             (000, 255, 000),           # Green
             (255, 000, 000),           # Blue
             (255, 255, 000),           # Yellow
             (000, 255, 255),           # Cyan
             (255, 000, 255),           # Magenta
             (255, 128, 000),           # Orange
             (000, 255, 128),
             (128, 000, 255),
             (255, 000, 128),
             (128, 255, 000),
             (000, 128, 255))

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
    MatePiece = 9

    # MISC TASKS
    InitAruco = 10
    MoveWeightAway = 11
    Celebrate = 12

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
        for i, (task, taskdata) in enumerate(reversed(list(zip(self.tasks[:5], self.taskdata[:5])))):
            s += f"{i+1}. " + str(task)
            if taskdata:
                s += " | Task Data: " + str(taskdata)
            s += "\n"
        if len(self.tasks) > 5:
            s += f"... {len(self.tasks) - 5} more items on the stack not shown\n."
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
        # self.vision = vision.VisionMatcher('/home/me134/me134ws/src/HW1/gunter_exploded.jpg')

        # Stack
        self.tasks = TaskStack()
        self.tasks.push(SolverTask.Celebrate)
        self.tasks.push(SolverTask.MoveWeightAway)
        self.tasks.push(SolverTask.PutPiecesTogether)
        self.tasks.push(SolverTask.SeparatePieces)
        self.tasks.push(SolverTask.InitAruco)
        self.tasks.push(SolverTask.GetView)

        # Series of actions to perform
        self.action_queue = []

        self.puzzle_grid = PuzzleGrid(offset = np.array(self.get_puzzle_region()[0:2]) + [100, 60])
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

            # self.pieces_solved += 1
            # if self.pieces_solved == self.num_pieces:
            rospy.loginfo(f"[Solver] Solved all {self.num_pieces} pieces! All done.")
            return

            # Get a view of the available pieces
            # self.tasks.append(SolverTask.PutPiecesTogether)
            # self.tasks.append(SolverTask.GetViewCleared, task_data={'merge': False})

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
        elif curr_task == SolverTask.MoveWeightAway:
            status.assert_ok()
        elif curr_task == SolverTask.Celebrate:
            status.assert_ok()
        else:
            raise RuntimeError(f"Unknown task: {curr_task}")

    def apply_next_action(self, controller):
        '''
        This function will continuously apply the action at the top of the stack.
        To break out of the recursive loop, simply "return".
        '''

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
                controller.reset(location='puzzle_region')
                return

        elif curr_task == SolverTask.InitAruco:
            self.detector.init_aruco()
            self.tasks.pop()
        
        elif curr_task == SolverTask.MoveWeightAway:
            self.tasks.pop()

            weight_origin = self.detector.find_aruco(4)
            print(weight_origin)
            weight_dest = np.array([1400, 900]).astype(np.float32)
            self.tasks.push(SolverTask.PlacePiece, task_data={'jiggle': False, 'height': .01})
            self.tasks.push(SolverTask.MoveArm, task_data={'dest': weight_dest, 'height': .10})
            self.tasks.push(SolverTask.LiftPiece, task_data={'height': .007})
            self.tasks.push(SolverTask.MoveArm, task_data={'dest': weight_origin, 'height': .10})

        elif curr_task == SolverTask.Celebrate:
            controller.celebrate()
            return

        elif curr_task == SolverTask.MoveArm:
            controller.move_to_pixelcoords(task_data['dest'], task_data.get('turn', 0), task_data.get('height', .05))
            return

        elif curr_task == SolverTask.LiftPiece:
            controller.lift_piece(careful=task_data.get('careful', True),
                height=task_data.get('height', -.005))
            return

        elif curr_task == SolverTask.PlacePiece:
            controller.place_piece(jiggle=task_data['jiggle'], 
                careful=task_data.get('careful', True),
                height=task_data.get('height', -.005),
                tightening=task_data.get('tightening', np.array([0, 0])))
            return

        elif curr_task == SolverTask.MatePiece:
            # Matches the two biggest pieces
            def drawSide(image, side, color):
                for x,y in np.array(side).reshape(-1,2):
                    image[y,x] = color

            def drawSides(image, sides, color=None):
                for i in range(len(sides)):
                    c = color if color else COLORLIST[i % len(COLORLIST)]
                    drawSide(image, sides[i], c)

            # Makes first piece the main puzzle
            self.tasks.pop()
            pieces = sorted(self.detector.pieces, key=lambda x: x.xmin + x.ymin)

            if len(pieces) < 2:
                rospy.logwarn("[Solver]: Detected less than 2 pieces while mating, aborting.")
                self.tasks.pop()
                return self.apply_next_action(controller)

            solved_part = pieces[0]
            new_piece = pieces[1]
            # Uses assumption that puzzle is starting from top left to figure out which piece is which
            loc1, loc2 = np.array(pieces[0].get_location()), np.array(pieces[1].get_location())
            delta = loc1 - loc2
            delta = delta / np.linalg.norm(delta)

            def find_weight_dest():
                return loc1 + delta * 40

            self.puzzle_grid.piece = (pieces[0])

            neighbors_rel = task_data.get('neighbors')
            grid_loc  = task_data.get('grid_loc')
            piece_size = np.array([135, 115])
            neighbor_mate = np.array(neighbors_rel[0])
            neighbor_vec = np.array(neighbors_rel[0]) - grid_loc
            puzzle_edge = self.puzzle_grid.grid_to_pixel(neighbor_mate) - piece_size * neighbor_vec/2
            piece_edge  = loc2 + piece_size * neighbor_vec/2
            
            alignment_offset = np.array([0, 0], dtype=int)
            if np.all(grid_loc == (0, 1)):
                alignment_offset = np.array([0, 6], dtype=int)

            # Cropping the left side of the puzzle if puzzle is too big
            # done_cols = 0
            # for col in range(5):
            #     if np.all(self.puzzle_grid.occupied[col, :] == 0):
            #         done_cols += 1
            #     else:
            #         break
            # piece_width = 135
            # piece_height = 115
            # solved_part.mask[solved_part.ymin:solved_part.ymin + 4*piece_height, \
            #                 solved_part.xmin:solved_part.xmax + done_cols*piece_width] = 0
            # solved_part.update_mask()

            dx, dy, dtheta = Solver.get_mating_correction(solved_part, new_piece, puzzle_edge, piece_edge)

            print("Piece sizes: ", np.sum(pieces[0].mask), np.sum(pieces[1].mask))
            
            # dx, dy, dtheta, side0, side1 = pieces[1].find_contour_match(pieces[0], match_threshold=20, return_sides=True)
            
            self.puzzle_grid.occupied[grid_loc[0], grid_loc[1]] = 1
            # for normal puzzle: height = -.001
            self.tasks.push(SolverTask.PlacePiece, task_data={'jiggle': True, 'height': 0.000, 'tightening': -np.array(grid_loc) * 0.002})
            self.tasks.push(SolverTask.MoveArm, task_data={'dest': pieces[1].get_pickup_point() + np.array([dx, dy]) + alignment_offset, 'turn': dtheta})
            self.tasks.push(SolverTask.LiftPiece)
            self.tasks.push(SolverTask.MoveArm, task_data={'dest': pieces[1].get_pickup_point()})
            if dx != 0 or dy != 0:
                weight_origin = self.detector.find_aruco(4)
                weight_dest = find_weight_dest()
                weight_distance = np.linalg.norm(weight_dest - weight_origin)
                rospy.loginfo(f"[Solver] Weight is {weight_distance} away")
                # Don't bother moving the weight if it is close enough
                if weight_distance > 50 and solved_part.area < 15000*5:
                    # Move the weight to the main puzzle
                    self.tasks.push(SolverTask.PlacePiece, task_data={'jiggle': False, 'height': .01})
                    self.tasks.push(SolverTask.MoveArm, task_data={'dest': weight_dest, 'height': .10})
                    self.tasks.push(SolverTask.LiftPiece, task_data={'height': .007})
                    self.tasks.push(SolverTask.MoveArm, task_data={'dest': self.detector.find_aruco(4), 'height': .10})
            plan_img = np.zeros((1080, 1720, 3), dtype=np.uint8) + 128
            
            # drawSide(plan_img, side0 + [pieces[0].xmin, pieces[0].ymin], [255, 0, 0])
            # drawSide(plan_img, side1 + [pieces[1].xmin, pieces[1].ymin], [255, 0, 0])
            #drawSides
            dummy_piece = pieces[1].copy()
            dummy_piece.rotate(dtheta)
            dummy_piece.move_to(np.array(dummy_piece.get_location()[0]) + dx, np.array(dummy_piece.get_location()[1]) + dy)
            plan_img[pieces[0].mask.astype(bool)] += np.array(pieces[0].color).astype(np.uint8)
            plan_img[dummy_piece.mask.astype(bool)] += np.array(pieces[1].color).astype(np.uint8)
            cv2.circle(plan_img, tuple(puzzle_edge.astype(int)), 15, (255, 255, 255), -3)
            cv2.circle(plan_img, tuple(piece_edge.astype(int)), 15, (255, 255, 255), -3)
            self.pub_contour_matching.publish(self.detector.bridge.cv2_to_imgmsg(plan_img, "bgr8"))

        # HIGH-LEVEL TASKS
        elif curr_task == SolverTask.SeparatePieces:
            # We are trying to clear the center of the playground to make room for
            # the completed puzzle. At the same time, we are rotating all the pieces
            # to their correct orientation.

            move_to_puzzle_region = False

            for piece in sorted(self.piece_list, key=lambda piece: piece.xmin):
                if not piece.is_valid():
                    continue
                if piece.on_border_of_region(self.get_screen_region()):
                    continue # deal with it later

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
                # Once all pieces are not in the puzzle region, check whether any pieces are on the
                # border of the screen, as that is bad for the solver.
                for piece in self.piece_list:
                    if piece.on_border_of_region(self.get_screen_region()):
                        move_to_puzzle_region = True
                        rotation_offset = 0
                        break
                else:
                    # All pieces have been cleared from the puzzle region
                    rospy.loginfo("[Solver] No pieces left to separate, continuing to next solver task.")

                    # TODO: in the future, we might want to not pop, so that
                    # we can recover from errors.
                    self.tasks.pop()
                    return self.apply_next_action(controller)

            piece_origin = piece.get_pickup_point()
            piece_destination = self.find_available_piece_spot(piece, rotation_offset, move_to_puzzle_region)
            piece_destination = piece.correct_point_by_pickup_offset(piece_destination)

            # Add in reverse
            self.tasks.push(SolverTask.GetView)
            self.tasks.push(SolverTask.PlacePiece, task_data={'jiggle': False, 'careful': False})
            self.tasks.push(SolverTask.MoveArm, task_data={'dest': piece_destination, 'turn': rotation_offset})
            self.tasks.push(SolverTask.LiftPiece, task_data={'careful': False})
            self.tasks.push(SolverTask.MoveArm, task_data={'dest': piece_origin})

            # Continue onto the next action
            return self.apply_next_action(controller)

        elif curr_task == SolverTask.PutPiecesTogether:
            # All pieces should now be on the border and oriented orthogonal.
            # Select pieces from the border and put them in the right place

            self.tasks.pop()

            locations, rots = self.mask_match(self.piece_list)
            hackystack = TaskStack() # temporary hacky solution
            temp_grid = PuzzleGrid(offset = np.array(self.get_puzzle_region()[0:2]) + [100, 60])

            plan_img = self.detector.latestImage.copy()

            # Mark out puzzle territory in green
            plan_img[self.get_puzzle_region_as_slice()] += np.array([0, 40, 0], dtype=np.uint8)

            def place_offset(location, offset=60):
                print(temp_grid.get_neighbors(location), location)
                neighbors = temp_grid.get_neighbors(location) - location
                return -neighbors.sum(axis=0) * offset

            def processpiece(i, first=False):
                offset = place_offset(locations[i]) if not first else 0
                # offset = 0
                loc = temp_grid.grid_to_pixel(locations[i])
                piece = self.piece_list[i]
                loc = np.array(piece.correct_point_by_pickup_offset(loc))
                # Color selected piece
                color = np.array(piece.color).astype(np.uint8)
                # plan_img[piece.bounds_slice()] += piece.color.astype(np.uint8)
                dummy_piece = piece.copy()
                dummy_piece.rotate(rots[i])
                dummy_piece.move_to(loc[0], loc[1])
                cv2.circle(plan_img, (piece.x_center, piece.y_center), 15, piece.get_color(), -1)
                plan_img[dummy_piece.mask.astype(bool)] = color

                # NOT pushed in reverse because hackystack gets added to self.tasks in reverse
                hackystack.push(SolverTask.MoveArm, task_data={'dest': self.piece_list[i].get_pickup_point()})
                hackystack.push(SolverTask.LiftPiece, task_data={'careful': first})
                hackystack.push(SolverTask.MoveArm, task_data={'dest': loc + offset, 'turn': rots[i]})
                hackystack.push(SolverTask.PlacePiece, task_data={'jiggle': False, 'careful': False})
                if not first:
                    hackystack.push(SolverTask.GetViewPuzzle, task_data={'merge': False})
                    hackystack.push(SolverTask.MatePiece, task_data={'neighbors':temp_grid.get_neighbors(locations[i]), 'grid_loc': locations[i]})

                # self.action_queue.append(call_me(self.piece_list[i].get_location(), loc, turn = rots[i]*np.pi/2, jiggle=False))
                temp_grid.occupied[tuple(locations[i])] = 1
            # Find top left
            target_loc = [0, 0]
            for i in range(len(self.piece_list)):
                if np.array_equal(locations[i], target_loc):
                    break
            processpiece(i, first=True)
            loc_list = list()
            for i in range(len(self.piece_list)):
                loc_list.append([i, locations[i]])
            list.sort(loc_list, key=lambda x: np.linalg.norm(x[1]-target_loc))
            done = False
            while not done:
                done = True
                # for i in range(1):
                for i in range(len(self.piece_list)):
                    # Puts pieces in an order where they mate
                    if (temp_grid.occupied[tuple(loc_list[i][1])] == 0 and \
                        temp_grid.does_mate(loc_list[i][1])):
                        processpiece(loc_list[i][0])
                        done = False

            while len(hackystack) > 0:
                task, task_data = hackystack.pop()
                self.tasks.push(task, task_data=task_data)

            self.pub_clearing_plan.publish(self.detector.bridge.cv2_to_imgmsg(plan_img, "bgr8"))

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

    def get_region_of_instant_and_inescapable_death(self):
        return self.detector.death_region()
    def get_slice_of_instant_and_inescapable_death(self):
        xmin, ymin, xmax, ymax = self.get_region_of_instant_and_inescapable_death()
        return (slice(ymin, ymax), slice(xmin, xmax))

    def get_puzzle_region(self):
        '''
        Return (xmin, ymin, xmax, ymax) in pixel space
        representing the region where we want the solved puzzle to end up.
        '''
        # TODO
        return (830, 250, 1680, 850)

    def get_puzzle_keepouts(self):
        '''
        Return (xmin, ymin, xmax, ymax) in pixel space
        representing the region where we want the solved puzzle to end up.
        '''

        return 0

    def get_screen_region(self):
        return (0, 0, 1720, 1080)

    def get_puzzle_region_as_slice(self):
        xmin, ymin, xmax, ymax = self.get_puzzle_region()
        return (slice(ymin, ymax), slice(xmin, xmax))

    def get_aruco_regions(self):
        xmax = np.max(self.detector.aruco_corners_pixel[:, :, 0], axis=1).reshape(5, 1)
        ymax = np.max(self.detector.aruco_corners_pixel[:, :, 1], axis=1).reshape(5, 1)
        xmin = np.min(self.detector.aruco_corners_pixel[:, :, 0], axis=1).reshape(5, 1)
        ymin = np.min(self.detector.aruco_corners_pixel[:, :, 1], axis=1).reshape(5, 1)
        return np.hstack((xmin, ymin, xmax, ymax)).astype(int)

    def find_available_piece_spot(self, piece, rotation, move_to_puzzle_region=False):
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
        start_x, start_y = 50, 50
        for x in range(start_x, 1720-100, 10):
            for y in range(start_y, 1080-100, 10):
                dummy_piece.move_to_no_mask(x, y)

                # Publish our plans
                # plan_img = self.detector.latestImage.copy()
                # plan_img[self.get_puzzle_region_as_slice()] += np.array([0, 0, 40], dtype=np.uint8)
                # plan_img[piece.bounds_slice()] += np.array([40, 0, 0], dtype=np.uint8)
                # plan_img[dummy_piece.bounds_slice()] += np.array([0, 20, 20], dtype=np.uint8)
                # self.pub_clearing_plan.publish(self.detector.bridge.cv2_to_imgmsg(plan_img, "bgr8"))

                if not dummy_piece.fully_contained_in_region(self.get_screen_region(), buffer=20):
                    continue

                # Piece cannot go to the area we designate for the solved puzzle
                if dummy_piece.overlaps_with_region(self.get_puzzle_region()) and not move_to_puzzle_region:
                    continue

                if dummy_piece.overlaps_with_region(self.get_region_of_instant_and_inescapable_death()):
                    break # assume region is at the bottom

                # This is set when the piece is close to the border
                # Move it to the puzzle region where it will be fully seen
                # (it will later be moved away)
                if move_to_puzzle_region:
                    if not dummy_piece.fully_contained_in_region(self.get_puzzle_region(), buffer=50):
                        continue

                # Piece cannot go to the aruco areas
                flag = False
                for region in self.get_aruco_regions():
                    if dummy_piece.overlaps_with_region(region, buffer=20):
                        flag = True
                        break 
                if flag:
                    continue

                # Piece can only go to where there are no other pieces already
                elif np.all(free_space[dummy_piece.bounds_slice(padding=15)]):
                    dummy_piece = dummy_piece_copy.copy()
                    dummy_piece.move_to(x, y)

                    plan_img = self.detector.latestImage.copy()

                    # Mark out puzzle territory in red
                    plan_img[self.get_puzzle_region_as_slice()] += np.array([0, 0, 40], dtype=np.uint8)
                    
                    # Mark out death region in red
                    plan_img[self.get_slice_of_instant_and_inescapable_death()] = np.array([0, 0, 40], dtype=np.uint8)

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

    def color_similarity(self, sol_piece, piece):
        masked_piece = piece.natural_img * piece.thomas_mask.reshape(list(piece.thomas_mask.shape) + [1])
        masked_sol = sol_piece.natural_img * sol_piece.thomas_mask.reshape(list(sol_piece.thomas_mask.shape) + [1])

        s_mean_color1 = masked_sol[:masked_sol.shape[0]//2, :].reshape(-1, 3)
        s_mean_color1 = np.nanmean(s_mean_color1[s_mean_color1 > 0], axis = 0)
        s_mean_color2 = masked_sol[masked_sol.shape[0]//2:, :].reshape(-1, 3)
        s_mean_color2 = np.nanmean(s_mean_color2[s_mean_color2 > 0], axis = 0)

        p_mean_color1 = masked_piece[:masked_piece.shape[0]//2, :].reshape(-1, 3)
        p_mean_color1 = np.nanmean(p_mean_color1[p_mean_color1 > 0], axis = 0)
        p_mean_color2 = masked_piece[masked_piece.shape[0]//2:, :].reshape(-1, 3)
        p_mean_color2 = np.nanmean(p_mean_color2[p_mean_color2 > 0], axis = 0)

        smape1 = 2*abs(np.nanmean(s_mean_color1 - p_mean_color1)) / (0.001+np.nanmean(s_mean_color1) + np.nanmean(p_mean_color1.mean()))
        smape2 = 2*abs(np.nanmean(s_mean_color2 - p_mean_color2)) / (0.001+np.nanmean(s_mean_color2) + np.nanmean(p_mean_color2.mean()))
        score = -0.01 * (smape1 + smape2) 
        if (np.isnan(score)):
            score = 0
        return score

    def mask_match(self, pieces):
        # auto-detect
        ref = cv2.imread('/home/me134/me134ws/src/HW1/done_exploded_colored4.jpg')
        assert ref is not None
        detector = IlyaDetector()
        detector.process(ref)
        sol_pieces = detector.pieces.copy()

        assert len(pieces) == len(sol_pieces)

        matching = np.zeros((len(pieces), len(sol_pieces)))
        rotations = np.zeros((len(pieces), len(sol_pieces)))

        def normalized(piece):
            piece = piece.copy()
            rot = -piece.get_aligning_rotation()
            piece.move_to(200, 200)
            piece.mask = piece.mask[:400, :400]
            piece.rotate(rot)
            piece.mask = piece.mask / np.linalg.norm(piece.mask)
            
            return piece, rot

        def color_profile(piece):
            color_mask = piece.natural_img * piece.thomas_mask.reshape((piece.natural_img.shape[0], piece.natural_img.shape[1], 1))
            color_mask = color_mask.reshape((-1, 3)).astype(np.float32)
            nonzero = np.any(color_mask, axis=1)
            color_mask[nonzero] -= np.mean(color_mask)
            color_mask[nonzero] /= np.std(color_mask)
            colors = np.mean(color_mask[nonzero], axis=0)
            # print(colors)
            return colors
        piece_colors = np.zeros((len(pieces), 3))
        sol_piece_colors = np.zeros((len(pieces), 3))

        pieces_normalized = list()
        for i, piece in enumerate(pieces):
            normalized_piece, rot = normalized(piece)
            rotations[i, :] += rot
            pieces_normalized.append(normalized_piece)
            piece_colors[i] = color_profile(piece)

        for j, sol_piece in enumerate(sol_pieces):
            sol_piece, rot = normalized(sol_piece)
            rotations[:, j] += -rot
            sol_piece_colors[j] = color_profile(sol_piece)
            print(f"Progress: {j}/{len(sol_pieces)}")
            for i, piece in enumerate(pieces_normalized):
                piece = piece.copy()
                matching[i, j] = 0
                best_rot = 0
                color_score = np.sum(piece_colors[i] * sol_piece_colors[j]) * 0.2
                for dtheta in range(4):
                    piece.mask = piece.mask / np.linalg.norm(piece.mask)
                    dot = np.sum(sol_piece.mask * piece.mask)

                    alpha = 0.9
                    total_score = alpha * dot + (1 - alpha) * color_score

                    if total_score > matching[i, j]:
                        matching[i, j] = total_score# + self.color_similarity(sol_piece, piece)
                        best_rot = (dtheta * np.pi / 2)
                    piece.rotate(np.pi/2)
                rotations[i, j] += best_rot

        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(-matching)

        fake_grid = np.zeros((1080, 1720))
        positions = np.zeros((len(pieces), 2))
        rots = np.zeros(len(pieces))

        for i, j in zip(row_ind, col_ind):
            piece = pieces[i]
            sol_piece = sol_pieces[j]
            print(f"[Solver]: Piece {i} similarity: {matching[i, j]}")
            positions[i] = np.array([sol_piece.x_center, sol_piece.y_center])
            fake_piece = piece.copy()
            fake_piece.move_to(positions[i, 0], positions[i, 1])
            fake_piece.rotate(rotations[i, j])
            rots[i] = rotations[i, j]

            fake_grid += fake_piece.mask
        
        print(f"[Solver]: Average similarity: {matching[row_ind, col_ind].sum()/len(pieces)}") 
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # grid_locs = np.zeros((5, 4))
        grid_locs = np.zeros((20, 2)).astype(int)
        locs = sorted(list(positions), key = lambda pos: pos[0] + pos[1] * 9)
        count = 0
        for j in range(4):
            for i in range(5):
                for index in range(len(pieces)):
                    if np.array_equal(positions[index], locs[count]):
                        grid_locs[index] = [i, j]
                        count += 1
                        break
                else:
                    raise RuntimeError("Shouldn't be reached! Could be an incorrect number of pieces"
                    + f"Expects 20, found {len(pieces)}")

        return grid_locs, rots

    @staticmethod
    def get_mating_correction(fixed_piece, new_piece, solved_edge_loc, new_edge_loc):
    
        def get_edge(piece, edge_loc):
            sides = piece.get_sides()
            best_score = np.inf
            best_side = None
            for side in sides:
                score = np.linalg.norm(np.mean(side, axis=0) - edge_loc)
                if score < best_score:
                    best_side = side
                    best_score = score
            return best_side

        solved_edge = get_edge(fixed_piece, solved_edge_loc)
        new_edge = get_edge(new_piece, new_edge_loc)

        new_edge_to_solved_edge = np.mean(solved_edge, axis=0) - np.mean(new_edge, axis=0)
        
        new_edge_moved = (new_edge + new_edge_to_solved_edge).astype(int)

        rotation_center = np.array(new_piece.get_pickup_point()).astype(np.float32)

        fixed_edge_arr = np.zeros((2000, 2000)).astype(np.uint8)
        new_edge_arr = np.zeros((2000, 2000)).astype(np.uint8)
        x, y = tuple(np.mean(new_edge, axis=0).astype(int))

        for p in solved_edge:
            fixed_edge_arr[p[1], p[0]] = 1
        for p in new_edge_moved:
            new_edge_arr[p[1], p[0]] = 1

        fixed_edge_arr = cv2.dilate(fixed_edge_arr, None, iterations=5).astype(np.float32)
        new_edge_arr = cv2.dilate(new_edge_arr, None, iterations=5).astype(np.float32)

        fixed_edge_arr /= np.linalg.norm(fixed_edge_arr)

        def fun(arr):
            dx = int(arr[0])
            dy = int(arr[1])
            dtheta = arr[2]
            dtheta_deg = dtheta*180/np.pi
            corrected = new_edge_arr.copy()
            corrected[y-100+dy:y+100+dy, x-100-dx:x+100-dx] = corrected[y-100:y+100, x-100:x+100]
            dtheta = np.radians(dtheta_deg)

            center = rotation_center + np.float32([dx, dy])

            rotate_matrix = cv2.getRotationMatrix2D(center=tuple(center), angle=dtheta_deg, scale=1)
            corrected = cv2.warpAffine(corrected, rotate_matrix, corrected.shape[::-1])

            corrected /= np.linalg.norm(corrected)
            score = np.sum(corrected * fixed_edge_arr)
            return -score


        x0 = np.array([-1, -1, -0.01])
        from scipy.optimize import minimize
        res = minimize(fun, x0, method='nelder-mead',
                    options={'xatol': 1e-8, 'disp': True, 'maxiter': 20},
                    bounds = [(-10, 10), (-10, 10), (-3, 3)])

        res_dx = int(res.x[0])
        res_dy = int(res.x[1])
        dtheta = res.x[2]
        
        dx = new_edge_to_solved_edge[0] + res_dx
        dy = new_edge_to_solved_edge[1] - res_dy
        
        return dx, dy, dtheta