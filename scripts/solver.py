#!/usr/bin/env python3

import enum

import rospy

import numpy as np
from scripts.actions import RobotAction

class SolverTask(enum.Enum):
    GetView = 1
    SeparatePieces = 2
    PutPiecesTogether = 3

class Solver:
    def __init__(self):
        pass

        # Stack
        self.tasks = []
        self.tasks.append(SolverTask.PutPiecesTogether)
        self.tasks.append(SolverTask.SeparatePieces)
        self.tasks.append(SolverTask.GetView)

        self.num_pieces = 20
        self.pieces_cleared = 0
        self.pieces_solved = 0

    # Public methods
    def notify_action_completed(self, status):
        if not self.tasks:
            raise RuntimeError("No current tasks")

        curr_task = self.tasks[-1]
        if curr_task == SolverTask.GetView:
            if status == "ok":
                self.tasks.pop()
            else:
                pass
        elif curr_task == SolverTask.SeparatePieces:

            pass

    def apply_next_action(self, controller):

        if not self.tasks:
            controller.idle()
            return

        curr_task = self.tasks[-1]
        if curr_task == SolverTask.GetView:
            # If we want to get the view, simply go to the reset position, where
            # the robot arm is not in the camera frame.
            controller.reset()
            return

        elif curr_task == SolverTask.SeparatePieces:
            # We are trying to clear the center of the playground to make room for
            # the completed puzzle. At the same time, we are rotating all the pieces
            # to their correct orientation.

            for piece in piece_list:
                # Find piece which is not rotated correctly or is in the center

                rotation_offset = self.get_rotation_offset(piece.img)
                threshold_rotation_error = 0.05
                if abs(rotation_offset) > threshold_rotation_error:
                    break
                if piece.overlaps_with_region(self.get_puzzle_region()):
                    break
            else:
                # Nothing more to do in the current task!
                self.tasks.pop()
                return self.apply_next_action(controller)

            piece_origin = piece.get_center()
            piece_destination = self.find_available_piece_spot()
            controller.move_piece(piece_origin, piece_destination)
            return

        elif curr_task == SolverTask.PutPiecesTogether:
            # All pieces should now be on the border and oriented correctly.
            # Select pieces from the border and put them in the right place

            target_piece = self.reference_pieces[self.pieces_solved]
            for piece in piece_list:
                if self.pieces_match(piece.img, target_piece.img):
                    break
            else:
                # Error, no piece found!
                rospy.logwarn(f"[Solver] Target piece #{self.peices_solved} not found! Getting a new view.")
                self.tasks.append(SolverTask.GetView)
                return self.apply_next_action(controller)

            piece_origin = piece.get_center()

            # FIXME, this isn't quite right but is a good start
            piece_destination = target_piece.get_center()

            controller.move_piece(piece_origin, piece_destination)
        else:
            raise RuntimeError(f"Unknown task: {curr_task}")


    # Private methods
    def get_rotation_offset(self, piece_img):
        '''
        Given an image of a piece, return how many radians clockwise it should be
        turned to be in the correct orientation.
        '''
        # TODO
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
        return (100, 100, 300, 300)

    def find_available_piece_spot(self):
        '''
        Return a location on the border of the playground which has enough free space
        to put a piece there.
        '''
        # TODO
        return (50, 50)