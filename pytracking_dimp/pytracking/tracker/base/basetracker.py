import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 as cv
import os
import time
import torch
from pytracking.utils.convert_vot_anno_to_rect import convert_vot_anno_to_rect
from pytracking.utils.visdom import Visdom
from pytracking.features.preprocessing import torch_to_numpy
from pytracking.utils.plotting import draw_figure

import numpy as np
try:
    # Only needed when running vot
    import pytracking.VOT.vot as vot
except:
    pass


#fast depth completion
from ip_basic import depth_map_utils

class BaseTracker:
    """Base class for all trackers."""
    def visdom_ui_handler(self, data):
        if data['event_type'] == 'KeyPress':
            if data['key'] == ' ':
                self.pause_mode = not self.pause_mode

            elif data['key'] == 'ArrowRight' and self.pause_mode:
                self.step = True

    def __init__(self, params):
        self.params = params

        self.pause_mode = False
        self.step = False

        self.visdom = None
        if self.params.debug > 0 and self.params.visdom_info.get('use_visdom', True):
            try:
                self.visdom = Visdom(self.params.debug, {'handler': self.visdom_ui_handler, 'win_id':  'Tracking'},
                                     visdom_info=self.params.visdom_info)

                # Show help
                help_text = 'You can pause/unpause the tracker by pressing ''space'' with the ''Tracking'' window ' \
                            'selected. During paused mode, you can track for one frame by pressing the right arrow key.' \
                            'To enable/disable plotting of a data block, tick/untick the corresponding entry in ' \
                            'block list.'
                self.visdom.register(help_text, 'text', 1, 'Help')
            except:
                time.sleep(0.5)
                print('!!! WARNING: Visdom could not start, so using matplotlib visualization instead !!!\n'
                      '!!! Start Visdom in a separate terminal window by typing \'visdom\' !!!')


    def initialize(self, image, info: dict) -> dict:
        """Overload this function in your tracker. This should initialize the model."""
        raise NotImplementedError


    def track(self, image) -> dict:
        """Overload this function in your tracker. This should track in the frame and update the model."""
        raise NotImplementedError


    def track_sequence(self, sequence):
        """Run tracker on a sequence."""

        output = {'target_bbox': [],
                  'time': [],
                  'scores': []}

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in tracker_out.keys():
                if key not in output:
                    raise RuntimeError('Unknown output from tracker.')
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if val is not None:
                    output[key].append(val)

        # Initialize
        image = self._read_image(sequence.frames[0])
        #['image'] (480, 640, 3)
        if hasattr(self.params, 'use_depth_channel'):
            if self.params.use_depth_channel:
                print('have %d depth frames'%(len(sequence.depth_frames)))
                depth = self._read_depth(sequence.depth_frames[0])

                #depth = depth/1000.0 #from mm to m
                depth = np.repeat(np.expand_dims(depth,axis=2),3,axis=2)#.astype(np.uint8)
                if image.shape[0]!=depth.shape[0]:
                    depth=depth[1:image.shape[0]+1,:,:]
                #image = np.concatenate((image,np.expand_dims(depth,axis=2)),axis=2).astype(np.uint8)
                #print(['image'],image.shape)
                #['image'] (480, 640, 4)
                #print(['depth', depth.shape, np.mean(depth, (0,1)), np.std(depth, (0,1))])
                #['depth', (480, 640), 22.48,19.60]


        if self.params.visualization and self.visdom is None:
            self.init_visualization()
            self.visualize(image[:,:,0:3], sequence.get('init_bbox'))

        start_time = time.time()


        if hasattr(self.params, 'use_depth_channel'):
            out= self.initialize(image, depth, sequence.init_info())
        else:
            out= self.initialize(image, sequence.init_info())

        if out is None:
            out = {}
        _store_outputs(out, {'target_bbox': sequence.get('init_bbox'),
                             'time': time.time() - start_time,
                             'scores':1.0})

        if self.visdom is not None:
            self.visdom.register((image, sequence.get('init_bbox')), 'Tracking', 1, 'Tracking')

        # Track
        ind_frame=0
        for frame in sequence.frames[1:]:
            ind_frame=ind_frame+1
            self.ind_frame=ind_frame
            while True:
                if not self.pause_mode:
                    break
                elif self.step:
                    self.step = False
                    break
                else:
                    time.sleep(0.1)

            image = self._read_image(frame)

            if hasattr(self.params, 'use_depth_channel'):
                #print(['depth image',sequence.depth_frames[ind_frame]])
                depth = self._read_depth(sequence.depth_frames[ind_frame])
                #depth = depth/1000.0 #from mm to m
                depth = np.repeat(np.expand_dims(depth,axis=2),3,axis=2)#.astype(np.uint8)



            start_time = time.time()
            if hasattr(self.params, 'use_depth_channel'):
                out = self.track(image, depth)
            else:
                out = self.track(image)
            _store_outputs(out, {'time': time.time() - start_time, 'scores': self.debug_info['max_score']})


            #get gt_state if the gt_state for the whole sequence is provided
            if sequence.ground_truth_rect.shape[0]>1:
                self.gt_state=sequence.ground_truth_rect[ind_frame]

            if self.visdom is not None:
                self.visdom.register((image, out['target_bbox']), 'Tracking', 1, 'Tracking')
            elif self.params.visualization:
                # if hasattr(self.params, 'use_depth_channel'):
                #     self.visualize(image, out['target_bbox'], out_rgb['target_bbox'], out_depth['target_bbox'])
                # else:
                self.visualize(image, out['target_bbox'])

                #visualize the depth
                if hasattr(self.params, 'use_depth_channel'):
                    if os.path.exists(sequence.depth_frames[ind_frame]):
                        #dimage=self._read_image(sequence.depth_frames[ind_frame])
                        self.visualize_depth(np.uint8(255*depth/np.max(depth)),out['target_bbox'])
                        #print(depth.shape)
                        pass


        return output

    def track_videofile(self, videofilepath, optional_box=None):
        """Run track with a video file input."""

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"

        if hasattr(self, 'initialize_features'):
            self.initialize_features()

        cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + self.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        success, frame = cap.read()
        cv.imshow(display_name, frame)
        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, list, tuple)
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            self.initialize(frame, {'init_bbox': optional_box})
        else:
            while True:
                # cv.waitKey()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                           1.5, (0, 0, 0), 1)

                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                self.initialize(frame, {'init_bbox': init_state})
                break

        while True:
            ret, frame = cap.read()

            if frame is None:
                return

            frame_disp = frame.copy()

            # Draw box
            out = self.track(frame)
            state = [int(s) for s in out['target_bbox']]
            cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)

            font_color = (0, 0, 0)
            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)

            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ret, frame = cap.read()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                           (0, 0, 0), 1)

                cv.imshow(display_name, frame_disp)
                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                self.initialize(frame, {'init_bbox': init_state})

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

    def track_webcam(self):
        """Run tracker with webcam."""

        class UIControl:
            def __init__(self):
                self.mode = 'init'  # init, select, track
                self.target_tl = (-1, -1)
                self.target_br = (-1, -1)
                self.mode_switch = False

            def mouse_callback(self, event, x, y, flags, param):
                if event == cv.EVENT_LBUTTONDOWN and self.mode == 'init':
                    self.target_tl = (x, y)
                    self.target_br = (x, y)
                    self.mode = 'select'
                    self.mode_switch = True
                elif event == cv.EVENT_MOUSEMOVE and self.mode == 'select':
                    self.target_br = (x, y)
                elif event == cv.EVENT_LBUTTONDOWN and self.mode == 'select':
                    self.target_br = (x, y)
                    self.mode = 'track'
                    self.mode_switch = True

            def get_tl(self):
                return self.target_tl if self.target_tl[0] < self.target_br[0] else self.target_br

            def get_br(self):
                return self.target_br if self.target_tl[0] < self.target_br[0] else self.target_tl

            def get_bb(self):
                tl = self.get_tl()
                br = self.get_br()

                bb = [min(tl[0], br[0]), min(tl[1], br[1]), abs(br[0] - tl[0]), abs(br[1] - tl[1])]
                return bb

        ui_control = UIControl()
        cap = cv.VideoCapture(0)
        display_name = 'Display: ' + self.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        cv.setMouseCallback(display_name, ui_control.mouse_callback)

        if hasattr(self, 'initialize_features'):
            self.initialize_features()

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            frame_disp = frame.copy()

            if ui_control.mode == 'track' and ui_control.mode_switch:
                ui_control.mode_switch = False
                init_state = ui_control.get_bb()
                self.initialize(frame, {'init_bbox': init_state})

            # Draw box
            if ui_control.mode == 'select':
                cv.rectangle(frame_disp, ui_control.get_tl(), ui_control.get_br(), (255, 0, 0), 2)
            elif ui_control.mode == 'track':
                out = self.track(frame)
                state = [int(s) for s in out['target_bbox']]
                cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                             (0, 255, 0), 5)

            # Put text
            font_color = (0, 0, 0)
            if ui_control.mode == 'init' or ui_control.mode == 'select':
                cv.putText(frame_disp, 'Select target', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 1)
                cv.putText(frame_disp, 'Press q to quit', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
            elif ui_control.mode == 'track':
                cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
                cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
                cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ui_control.mode = 'init'

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

    def track_vot(self):
        """Run tracker on VOT."""
        def _convert_anno_to_list(vot_anno):
            vot_anno = [vot_anno[0][0][0], vot_anno[0][0][1], vot_anno[0][1][0], vot_anno[0][1][1],
                        vot_anno[0][2][0], vot_anno[0][2][1], vot_anno[0][3][0], vot_anno[0][3][1]]
            return vot_anno

        def _convert_image_path(image_path):
            image_path_new = image_path[20:- 2]
            return "".join(image_path_new)

        handle = vot.VOT("polygon")

        vot_anno_polygon = handle.region()
        vot_anno_polygon = _convert_anno_to_list(vot_anno_polygon)

        init_state = convert_vot_anno_to_rect(vot_anno_polygon, self.params.vot_anno_conversion_type)

        image_path = handle.frame()
        if not image_path:
            return
        image_path = _convert_image_path(image_path)

        image = self._read_image(image_path)
        self.initialize(image, {'init_bbox': init_state})

        if self.visdom is not None:
            self.visdom.register((image, init_state), 'Tracking', 1, 'Tracking')

        # Track
        while True:
            while True:
                if not self.pause_mode:
                    break
                elif self.step:
                    self.step = False
                    break
                else:
                    time.sleep(0.1)

            image_path = handle.frame()
            if not image_path:
                break
            image_path = _convert_image_path(image_path)

            image = self._read_image(image_path)
            out = self.track(image)
            state = out['target_bbox']

            if self.visdom is not None:
                self.visdom.register((image, state), 'Tracking', 1, 'Tracking')
            handle.report(vot.Rectangle(state[0], state[1], state[2], state[3]))

    def reset_tracker(self):
        pass

    def press(self, event):
        if event.key == 'p':
            self.pause_mode = not self.pause_mode
            print("Switching pause mode!")
        elif event.key == 'r':
            self.reset_tracker()
            print("Resetting target pos to gt!")

    def init_visualization(self):
        # plt.ion()
        self.pause_mode = False
        self.fig, self.ax = plt.subplots(1)
        self.fig2,self.ax2= plt.subplots(1)
        self.fig.canvas.mpl_connect('key_press_event', self.press)
        plt.tight_layout()

    def visualize(self, image, state, *var):
        self.ax.cla()
        self.ax.imshow(image)

        if (state[2]!=0 and state[3]!=0):
            self.ax.text(10, 30, 'FOUND', fontsize=14, bbox=dict(facecolor='green', alpha=0.2))
        else:
            self.ax.text(10, 30, 'NOT FOUND', fontsize=14,bbox=dict(facecolor='red', alpha=0.2))
            pass
        if len(var)==0:
            rect = patches.Rectangle((state[0], state[1]), state[2], state[3], linewidth=1, edgecolor='r', facecolor='none')
            self.ax.add_patch(rect)

        if len(var)>0:#state_rgb, state_depth, provided
            state_rgb  =var[0]
            state_depth=var[1]
            #draw one dot for the center of state_rgb

            # self.ax.plot(state_rgb[0]+state_rgb[2]/2, state_rgb[1]+state_rgb[3]/2,'ro')
            # self.ax.plot(state_depth[0]+state_depth[2]/2, state_depth[1]+state_depth[3]/2, 'bo')
            # self.ax.plot(state[0]+state[2]/2, state[1]+state[3]/2,'wo')

            #another dot for the center of state_depth
            # rect_rgb= patches.Rectangle((state_rgb[0], state_rgb[1]), state_rgb[2], state_rgb[3], linewidth=2, edgecolor='r', facecolor='none')
            # self.ax.add_patch(rect_rgb)
            rect_depth= patches.Rectangle((state_depth[0], state_depth[1]), state_depth[2], state_depth[3], linewidth=2, edgecolor='b', facecolor='none')
            self.ax.add_patch(rect_depth)
            rect = patches.Rectangle((state[0], state[1]), state[2], state[3], linewidth=1, edgecolor='w', facecolor='none')
            self.ax.add_patch(rect)
        #print(['var', var])
        #['var', (tensor([263.5000, 266.5000]), tensor([263.5000, 266.5000]), tensor([263.6045, 271.1568]))]


        if hasattr(self, 'gt_state') and True:
            gt_state = self.gt_state
            self.ax.plot(gt_state[0]+gt_state[2]/2, gt_state[1]+gt_state[3]/2, 'go')
            rect = patches.Rectangle((gt_state[0], gt_state[1]), gt_state[2], gt_state[3], linewidth=2, edgecolor='g', facecolor='none')
            self.ax.add_patch(rect)

        self.ax.set_axis_off()
        self.ax.axis('equal')
        draw_figure(self.fig)

        if hasattr(self,'ind_frame'):
            if os.path.exists('./tracking_results/imgs'):
                self.fig.savefig('./tracking_results/imgs/img_%d.png'%self.ind_frame)

        if self.pause_mode:
            keypress = False
            while not keypress:
                keypress = plt.waitforbuttonpress()

    def visualize_depth(self, image, state):
        self.ax2.cla()
        self.ax2.imshow(image)

        self.ax2.set_axis_off()
        self.ax2.axis('equal')
        plt.draw()
        plt.pause(0.001)

        if hasattr(self,'ind_frame'):
            if os.path.exists('./tracking_results/imgs'):
                self.fig2.savefig('./tracking_results/imgs/depth_%d.png'%self.ind_frame)

        if self.pause_mode:
            plt.waitforbuttonpress()


    def show_image(self, im, plot_name=None, ax=None):
        if isinstance(im, torch.Tensor):
            im = torch_to_numpy(im)
        # plot_id = sum([ord(x) for x in list(plot_name)])

        if ax is None:
            plot_fig_name = 'debug_fig_' + plot_name
            plot_ax_name = 'debug_ax_' + plot_name
            if not hasattr(self, plot_fig_name):
                fig, ax = plt.subplots(1)
                setattr(self, plot_fig_name, fig)
                setattr(self, plot_ax_name, ax)
                plt.tight_layout()
                ax.set_title(plot_name)
            else:
                fig = getattr(self, plot_fig_name, None)
                ax = getattr(self, plot_ax_name, None)

        ax.cla()
        ax.imshow(im)

        ax.set_axis_off()
        ax.axis('equal')
        ax.set_title(plot_name)
        draw_figure(fig)


    def _read_depth(self, image_file: str):

        # Full kernels
        FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
        FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
        FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
        FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
        FULL_KERNEL_31 = np.ones((31, 31), np.uint8)


        depth = cv.imread(image_file, cv.COLOR_BGR2GRAY)
        #print(['_read_depth', depth.min(), depth.max(), depth.mean(), depth.std()])
        if 'Princeton' in image_file: #depth.max()>=60000: # bug found, we need to bitshift depth.
            depth=np.bitwise_or(np.right_shift(depth,3),np.left_shift(depth,13))
        depth=depth/1000.0
        depth[depth>=8.0]=8.0
        depth[depth<=0.0]=8.0
        #depth=8.0-depth
        # Hole closing
        depth = cv.morphologyEx(depth, cv.MORPH_CLOSE, FULL_KERNEL_7)
        #depth = 255.0*depth/(np.max(depth)+1e-3)
        return depth

    def _read_image(self, image_file: str):
        return cv.cvtColor(cv.imread(image_file), cv.COLOR_BGR2RGB)
