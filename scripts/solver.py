#!/usr/bin/env python3

from ast import arg
import enum

import rospy

import cv2
from vision import cvt_color

from sensor_msgs.msg   import Image

import numpy as np

from thomas_detector import ThomasDetector, ThomasPuzzlePiece
from puzzle_grid import PuzzleGrid
import vision

class SolverTask(enum.Enum):
    InitAruco = 0
    GetView = 1
    SeparatePieces = 2
    PutPiecesTogether = 3
    SeparateOverlappingPieces = 4
    GetViewCleared = 5
    GetViewPuzzle = 6

class Status(enum.Enum):
    Ok = 0
    PieceDropped = 1

    def ok(self):
        return self == Status.Ok

class Solver:
    def __init__(self, detector):

        self.pub_clearing_plan = rospy.Publisher("/solver/clearing_plan",  Image, queue_size=1)
        self.vision = vision.VisionMatcher('/home/me134/me134ws/src/HW1/done_exploded_colored4.jpg')
        # Stack
        self.tasks = []
        self.tasks.append(SolverTask.PutPiecesTogether)
        self.tasks.append(SolverTask.SeparatePieces)
        self.tasks.append(SolverTask.InitAruco)
        self.tasks.append(SolverTask.GetView)

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
        if not self.tasks:
            rospy.logerror("[Solver] Action completed recieved but there are no current tasks!")

        curr_task = self.tasks[-1]
        if curr_task == SolverTask.GetView:
            # The camera should have a clear view of the pieces now.
            if status.ok():
                self.detector.snap()
                self.piece_list = self.detector.pieces.copy()
                self.tasks.pop()
            else:
                errmsg = f"[Solver] Unknown error: {status}"
                rospy.logerror(errmsg)
                raise RuntimeError(errmsg)

        elif curr_task == SolverTask.InitAruco:
            if status.ok():
                self.tasks.pop()
            else:
                errmsg = f"[Solver] Unknown error: {status}"
                rospy.logerror(errmsg)
                raise RuntimeError(errmsg)

        elif curr_task == SolverTask.SeparatePieces:
            if status.ok():
                self.pieces_cleared += 1
                if self.pieces_cleared == self.num_pieces:
                    rospy.loginfo(f"[Solver] Cleared all {self.num_pieces} pieces, moving on to next task.")
                    self.tasks.pop()

                # TEMP? Get a new view after every piece
                self.tasks.append(SolverTask.GetView)
            else:
                raise NotImplementedError()

        elif curr_task == SolverTask.PutPiecesTogether:
            if status.ok():
                self.pieces_solved += 1
                if self.pieces_solved == self.num_pieces:
                    rospy.loginfo(f"[Solver] Solved all {self.num_pieces} pieces! Moving on to next task.")
                    self.tasks.pop()
                # self.tasks.append(SolverTask.GetView)
            else:
                raise NotImplementedError()

        elif curr_task == SolverTask.SeparateOverlappingPieces:
             raise NotImplementedError()

        else:
            raise RuntimeError(f"Unknown task: {curr_task}")

    def apply_next_action(self, controller):

        if not self.tasks:
            rospy.loginfo("[Solver] No current tasks, sending Idle command.")
            controller.idle()
            return

        curr_task = self.tasks[-1]
        if curr_task == SolverTask.GetView:
            # If we want to get the view, simply go to the reset position, where
            # the robot arm is not in the camera frame.
            rospy.loginfo("[Solver] Current task is GetView, sending Reset command.")
            controller.reset()
            return
        elif curr_task == SolverTask.GetViewCleared:
            # If we want to get the view, simply go to the reset position, where
            # the robot arm is not in the camera frame.
            rospy.loginfo("[Solver] Current task is GetViewCleared, sending Reset command.")
            controller.reset()
            return
        elif curr_task == SolverTask.GetViewPuzzle:
            # If we want to get the view, simply go to the reset position, where
            # the robot arm is not in the camera frame.
            rospy.loginfo("[Solver] Current task is GetViewPuzzle, sending Reset command.")
            controller.reset()
            return

        elif curr_task == SolverTask.InitAruco:
            # If we want to get the view, simply go to the reset position, where
            # the robot arm is not in the camera frame.
            rospy.loginfo("[Solver] Current task is InitAruco, sending Reset command.")
            self.detector.init_aruco()
            return

        elif curr_task == SolverTask.SeparatePieces:
            # We are trying to clear the center of the playground to make room for
            # the completed puzzle. At the same time, we are rotating all the pieces
            # to their correct orientation.
            rospy.loginfo("[Solver] Current task is SeparatePieces, sending PieceMove command.")

            for piece in sorted(self.piece_list, key=lambda piece: piece.xmin):
                # Make sure piece is not already separated
                #if np.all(piece.get_center() > self.separated_loc - np.array([5, 5])):
                #    continue

                # Find piece which is not rotated correctly or is in the center
                rotation_offset = self.get_rotation_offset(piece.img)
                ### TEMP
                rotation_offset *= -1

                rotation_offset += np.pi / 4
                rotation_offset %= np.pi / 2
                rotation_offset -= np.pi / 4

                print("rotation offset: ", rotation_offset)
                ### END TEMP
                threshold_rotation_error = 0.175
                if abs(rotation_offset) > threshold_rotation_error:
                    break
                if piece.overlaps_with_region(self.get_puzzle_region()):
                    break
            else:
                # Nothing more to do in the current task!
                rospy.logwarn("[Solver] No pieces left to separate, continuing to next solver task.")
                self.tasks.pop()
                return self.apply_next_action(controller)

            piece_origin = piece.get_center()
            piece_destination = self.find_available_piece_spot(piece, rotation_offset)
            controller.move_piece(piece_origin, piece_destination, turn=rotation_offset, jiggle=False)
            return

        elif curr_task == SolverTask.PutPiecesTogether:
            # All pieces should now be on the border and oriented orthogonal.
            # Select pieces from the border and put them in the right place
            rospy.loginfo("[Solver] Current task is PutPiecesTogether, sending PieceMove command.")
            class call_me():
                def __init__(self, *args, **kwargs):
                    self.args = args
                    self.kwargs = kwargs
                def __call__(self):
                    controller.move_piece(*self.args, **self.kwargs)
            print(self.action_queue)
            if self.action_queue:
                f = self.action_queue.pop(0)
                f()
                return

            locations, rots, scores = self.vision.match_all(self.piece_list)
            loc = np.array([720, 350]) + self.puzzle_grid.grid_to_pixel(locations[0])
            controller.move_piece(self.piece_list[0].get_location(), loc, turn = rots[0]*np.pi/2, jiggle=False)
            self.puzzle_grid.occupied[tuple(locations[0])] = 1
            done = False
            while not done:
                done = True
                for i in range(1, len(self.piece_list)):
                    # Puts pieces in an order where they mate
                    if scores[i] < 99999 and self.puzzle_grid.occupied[tuple(locations[i])] == 0 and self.puzzle_grid.does_mate(locations[i]):
                        loc = np.array([720, 350]) + self.puzzle_grid.grid_to_pixel(locations[i])
                        # weight_destination = self.puzzle_grid.get_neighbor(locations[i])
                        # weight_pos_offset = np.array(weight_destination) - np.array(locations[i])
                        # weight_destination = self.puzzle_grid.get_pixel_from_grid(weight_destination)
                        # self.action_queue.append(call_me(controller.move_weight(weight_destination)))
                        self.action_queue.append(call_me(self.piece_list[i].get_location(), loc, turn = rots[i]*np.pi/2, jiggle=True))
                        self.puzzle_grid.occupied[tuple(locations[i])] = 1
                        done = False
            print(self.puzzle_grid.occupied.transpose())
            self.action_queue.append(self.tasks.pop)

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
            raise NotImplementedError()

        else:
            raise RuntimeError(f"Unknown task: {curr_task}")


    # Private methods
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

                # Piece can only go to where there are no other pieces already
                elif np.all(free_space[dummy_piece.bounds_slice(padding=20)]):
                    dummy_piece = dummy_piece_copy.copy()
                    dummy_piece.move_to(x, y)

                    plan_img = self.detector.latestImage.copy()

                    # Mark out puzzle territory in red
                    plan_img[self.get_puzzle_region_as_slice()] += np.array([0, 0, 40], dtype=np.uint8)

                    # Mark out spaces that aren't free in red
                    plan_img[free_space.astype(bool) == False] += np.array([0, 0, 40], dtype=np.uint8)

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
