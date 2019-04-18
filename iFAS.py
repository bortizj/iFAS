#!/usr/bin/env python2.7
# Importing necessary packages
import gi

gi.require_version('Gtk', '3.0')
gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
gi.require_version('GstPbutils', '1.0')
from gi.repository import GLib, Gtk, Gst, GObject, Pango, Gdk, GdkX11, GstVideo
import os, sys, pickle, subprocess, time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ['NO_AT_BRIDGE'] = str(1)

GLib.threads_init()
Gst.init(None)
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas
from scipy import ndimage
from scipy import misc
import my_utilities as MU
import my_widgets as MW
import optimization_tools as OT
import content_features as CF
import inspect
import importlib

# s = Gdk.Screen.get_default()
np.set_printoptions(precision=3)
# Create GUI Class and its components
line = np.linspace(0, 255, num=255, endpoint=True, retstep=False, dtype='uint8')
mat = np.rot90(np.matlib.repmat(line, 16, 1))
cmap = plt.get_cmap('jet')
misc.imsave('/tmp/color_map.png', cmap(mat))


class Main(object):
    def __init__(self):
        # Variable initialization
        self.working_path = os.path.dirname(os.path.abspath(__file__))
        self.available_packages = None  # IT is actually the set of modules to be call
        self.list_of_available_fidelity_groups = None  # IT is a string with the name of the packages
        self.get_list_of_packages("./list_of_packages")
        self.list_of_selected_methods = ['cd00_deltaE2000']
        self.list_of_methods = ['cd00_deltaE2000']
        self.plotted_cd = 'cd00_deltaE2000'
        self.x_axis = 'cd00_deltaE2000'
        self.y_axis = 'cd00_deltaE2000'
        self.package_name = 'cd_measures_pack'
        self.stopped = False
        self.start_time = 0
        self.selected_pacakage = self.available_packages[self.package_name]
        # self.videoheight_default = (s.get_height()-130)/2
        # self.videowidth_default = (s.get_width()-300)/2
        # Default settings
        self.multimedia_file_ref = self.working_path + '/sample_images/test_ref_0.bmp'
        self.image_ref = ndimage.imread(self.multimedia_file_ref)
        self.multimedia_file_pro = self.working_path + '/sample_images/test_pro_0.bmp'
        self.image_pro = ndimage.imread(self.multimedia_file_pro)
        self.cd = {}
        self.content_features = {}
        self.list_of_ref_samples = [self.multimedia_file_ref]
        self.list_of_test_samples = {}
        self.list_of_test_samples[self.multimedia_file_ref.split('/')[-1]] = [self.multimedia_file_pro]
        self.list_of_ref_samples = [self.multimedia_file_ref]
        self.cd[self.multimedia_file_ref.split('/')[-1]] = {}
        self.cd[self.multimedia_file_ref.split('/')[-1]][self.multimedia_file_pro.split('/')[-1]] = {}
        # Getting Window started with glade (See .glade File)
        self.builder = Gtk.Builder()
        self.window = self.builder.add_from_file('iFAS.glade')  # ,'VQAwindow'
        self.window = self.builder.get_object('MainWindow')
        self.window.set_title("iFAS - Image Fidelity Assessment Software")
        screen = self.window.get_screen()
        monitors = []
        for m in range(screen.get_n_monitors()):
            monitors.append(screen.get_monitor_geometry(m))
        curmon = screen.get_monitor_at_window(screen.get_active_window())
        monitor_par = monitors[curmon]
        self.videoheight_default = (monitor_par.height - 150) / 2
        self.videowidth_default = (monitor_par.width - 320) / 2
        self.window.move(monitor_par.x, monitor_par.y)
        # Comment/Uncomment the following two lines for standalone running
        self.window.connect("delete_event", lambda w, e: Gtk.main_quit())
        self.window.connect("destroy", lambda w: Gtk.main_quit())
        # Setting Image Reference
        # self.drawable_ref = self.builder.get_object("image1")
        drawable_loc = self.builder.get_object("hbox4")
        self.drawable_ref = Gtk.Image()
        self.set_image_on_scrolledwindow(drawable_loc, self.drawable_ref)
        # Setting Image Processed
        # self.drawable_pro = self.builder.get_object("image2")
        self.drawable_pro = Gtk.Image()
        self.set_image_on_scrolledwindow(drawable_loc, self.drawable_pro)
        # Setting Image Difference
        # self.drawable_dif = self.builder.get_object("image3")
        drawable_loc = self.builder.get_object("box5")
        self.drawable_dif = Gtk.Image()
        self.set_image_on_scrolledwindow(drawable_loc, self.drawable_dif)
        self.drawable_map = self.builder.get_object("image4")
        self.label_high = self.builder.get_object("label_high_val")
        self.label_high.modify_font(Pango.FontDescription('Sans 20'))
        self.label_low = self.builder.get_object("label_low_val")
        self.label_low.modify_font(Pango.FontDescription('Sans 20'))
        self.CD, _ = getattr(self.selected_pacakage, self.plotted_cd)(self.image_ref, self.image_pro)
        # Setting Plot canvas
        self.set_ploting_space()
        # Setting Text Viewer
        self.create_textview()
        # Default settings
        self.create_status_bar()
        MW.create_menu_bar(self)
        # Setting label
        self.label_current_frame = self.builder.get_object("label_current_image")
        self.label_current_frame.modify_font(Pango.FontDescription('Sans 20'))
        self.label_ref = self.builder.get_object("label3")
        self.label_ref.modify_font(Pango.FontDescription('Sans 20'))
        self.label_diff = self.builder.get_object("label_dif_image")
        self.label_diff.modify_font(Pango.FontDescription('Sans 20'))
        # Setting Start button
        self.button_start = self.builder.get_object("button1")
        self.button_start.connect("clicked", self.on_clicked)
        self.button_stop = self.builder.get_object("button2")
        self.button_stop.connect("clicked", self.on_stop_all)
        self.update_images()
        self.window.show_all()
        # self.window.maximize()
        self.on_about_click()
        self.update_screen_size()

    def on_stop_all(self, button=None):
        self.print_message("Stop Pressed")
        self.stopped = True

    def set_image_on_scrolledwindow(self, location, image):
        scrolledwindow = Gtk.ScrolledWindow()
        scrolledwindow.set_hexpand(True)
        scrolledwindow.set_vexpand(True)
        scrolledwindow.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolledwindow.set_property("width-request", self.videowidth_default)
        scrolledwindow.set_property("height-request", self.videoheight_default)
        scrolledwindow.add_with_viewport(image)
        location.pack_start(scrolledwindow, True, True, 0)

    def size_displaying_check(self):
        center_image = self.image_ref  # [diff_h:shape_image[0]-diff_h,diff_w:shape_image[1]-diff_w,:]
        misc.imsave('/tmp/temp_ref.png', center_image)
        center_image = self.image_pro  # [diff_h:shape_image[0]-diff_h,diff_w:shape_image[1]-diff_w,:]
        misc.imsave('/tmp/temp_pro.png', center_image)
        center_image = self.CD  # [diff_h:shape_image[0]-diff_h,diff_w:shape_image[1]-diff_w]
        cmap = plt.get_cmap('jet')
        center_image = (1. * center_image - np.min(center_image[:])) / (
        np.max(center_image[:]) - np.min(center_image[:]))
        center_image = np.delete(cmap(np.uint8(255 * center_image)), 3, 2)
        misc.imsave('/tmp/temp_cd.png', center_image)

    def update_images(self):
        self.histogram_imagediff()
        self.size_displaying_check()
        self.drawable_ref.set_from_file('/tmp/temp_ref.png')
        self.drawable_pro.set_from_file('/tmp/temp_pro.png')
        self.drawable_dif.set_from_file('/tmp/temp_cd.png')
        image1 = ndimage.imread('/tmp/temp_hist.png')
        image1 = np.rot90(image1)
        image1 = misc.imresize(image1, 1. * self.videoheight_default / image1.shape[0], interp='bilinear')
        image2 = ndimage.imread('/tmp/color_map.png')
        image2 = misc.imresize(image2, (image1.shape[0], 16), interp='bilinear')
        image3 = np.concatenate((image1[:, 0::2, :], image2), axis=1)
        misc.imsave('/tmp/temp_mapandhist.png', image3[8:-8, :, :])
        self.drawable_map.set_from_file('/tmp/temp_mapandhist.png')
        self.label_low.set_text('Min ' + format(np.min(self.CD[:]), '.3f'))
        self.label_high.set_text('Max ' + format(np.max(self.CD[:]), '.3f'))
        # Removing temp files
        os.remove("/tmp/temp_ref.png")
        os.remove("/tmp/temp_pro.png")
        os.remove("/tmp/temp_cd.png")
        os.remove("/tmp/temp_mapandhist.png")
        # os.remove("/tmp/color_map.png")
        os.remove("/tmp/temp_hist.png")
        self.update_screen_size()

    def set_ploting_space(self):
        # Setting Plot canvas
        self.figure = Figure()
        self.axis = self.figure.add_subplot(111)
        self.axis.tick_params(labelsize=8)
        self.axis.set_autoscale_on(True)
        box = self.axis.get_position()
        self.axis.set_position([box.x0 + 0.02, box.y0 + 0.05, box.width - 0.0, box.height])
        self.plotted, = self.axis.plot([], [], linestyle='None', marker='x', color='r', markersize=8, fillstyle='full',
                                       markeredgewidth=3.0)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.set_size_request(5. * self.videowidth_default / 8, self.videoheight_default)
        self.plot_place = self.builder.get_object("box4")
        self.plot_place.pack_start(self.canvas, True, True, 0)

    def create_textview(self):
        self.scrolledwindow = Gtk.ScrolledWindow()
        self.scrolledwindow.set_hexpand(True)
        self.scrolledwindow.set_vexpand(True)
        self.scrolledwindow.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.ALWAYS)
        self.scrolledwindow.set_property("width-request", 3. * self.videowidth_default / 8)
        self.scrolledwindow.set_property("height-request", self.videoheight_default)
        self.plot_place.pack_start(self.scrolledwindow, True, True, 0)

        self.textview = Gtk.TextView()
        self.textbuffer = self.textview.get_buffer()
        self.textview.connect("size-allocate", self.autoscroll)
        self.h_tag = self.textbuffer.create_tag("bold", size_points=12, weight=Pango.Weight.BOLD)
        self.scrolledwindow.add(self.textview)

    def autoscroll(self, *args):
        adj = self.textview.get_vadjustment()
        adj.set_value(adj.get_upper() - adj.get_page_size())

    def create_status_bar(self):
        self.status_bar = self.builder.get_object("levelbar1")
        self.status_bar.set_property("width-request", self.videowidth_default)
        self.status_bar.set_min_value(0)
        self.status_bar.set_max_value(100)
        self.status_bar.set_mode(Gtk.LevelBarMode.CONTINUOUS)

    def histogram_imagediff(self):
        figure_temp = Figure()
        if np.isnan(np.min(self.CD)):
            pass
        n, _, _ = plt.hist(MU.convert_vec(self.CD), np.linspace(np.min(self.CD[:]), np.max(self.CD[:]), 255 + 1))
        plt.xlim([np.nanmin(self.CD[:]), np.nanmax(self.CD[:])])
        plt.ylim([np.nanmin(n), np.nanmax(n)])
        plt.xticks([])
        plt.yticks([])
        plt.savefig('/tmp/temp_hist.png', bbox_inches='tight', facecolor='lightgray')
        plt.clf()
        plt.cla()
        plt.close(figure_temp)

    def on_fid_measures_click(self, button=None):
        self.list_of_methods = []
        for ii in range(len(inspect.getmembers(self.selected_pacakage, inspect.isfunction))):
            self.list_of_methods.append(inspect.getmembers(self.selected_pacakage, inspect.isfunction)[ii][0])
        popwin = MW.popup_window_with_list(self.list_of_methods, sel_method=Gtk.SelectionMode.MULTIPLE,
                                           message="Measure")
        self.list_of_selected_methods = popwin.list_items
        if not self.list_of_selected_methods:
            self.return_to_default()
        str_list_methods = ''
        str_list_methods += 'Reference ' + self.multimedia_file_ref.split('/')[-1] + '\n'
        str_list_methods += 'Sample ' + self.multimedia_file_pro.split('/')[-1] + '\n'
        for jj in self.list_of_selected_methods:
            if self.cd.has_key(self.multimedia_file_ref.split('/')[-1]):
                if self.cd[self.multimedia_file_ref.split('/')[-1]].has_key(self.multimedia_file_pro.split('/')[-1]):
                    if self.cd[self.multimedia_file_ref.split('/')[-1]][self.multimedia_file_pro.split('/')[-1]].has_key(jj):
                        str_list_methods += jj + ' = ' + "%.5f" % self.cd[self.multimedia_file_ref.split('/')[-1]] \
                            [self.multimedia_file_pro.split('/')[-1]][jj] + '\n'
            else:
                str_list_methods += jj + '\n'
        self.print_message('Current package: ' + self.package_name + '\n' + 'Selected methods:\n' + \
                           str_list_methods + 'Current differnce map: ' + self.plotted_cd)

    def on_clicked(self, button=None):
        self.stopped = False
        c = 1
        strin_of_values = ''
        if not self.cd.has_key(self.multimedia_file_ref.split('/')[-1]):
            self.cd[self.multimedia_file_ref.split('/')[-1]] = {}
        self.cd[self.multimedia_file_ref.split('/')[-1]][self.multimedia_file_pro.split('/')[-1]] = {}
        strin_of_values += 'Reference ' + self.multimedia_file_ref.split('/')[-1] + '\n'
        strin_of_values += 'Sample ' + self.multimedia_file_pro.split('/')[-1] + '\n'
        for ii in self.list_of_selected_methods:
            if hasattr(self.selected_pacakage, ii):
                self.CD, self.cd[self.multimedia_file_ref.split('/')[-1]][self.multimedia_file_pro.split('/')[-1]][ii] = \
                    getattr(self.selected_pacakage, ii)(self.image_ref, self.image_pro)
                strin_of_values += ii + ' = ' + "%.5f" % self.cd[self.multimedia_file_ref.split('/')[-1]] \
                    [self.multimedia_file_pro.split('/')[-1]][ii] + '\n'
            else:
                self.print_message('Error ' + self.pacakage_name + ' does not have a module' + ii + \
                                   '. Try again selecting the methods and packages first.')
            self.status_bar.set_value(100. * c / len(self.list_of_selected_methods))
            c += 1
            while Gtk.events_pending():
                Gtk.main_iteration()
            if self.stopped:
                return False
        self.plotted_cd = self.list_of_selected_methods[0]
        if hasattr(self.selected_pacakage, self.plotted_cd):
            self.CD, _ = getattr(self.selected_pacakage, self.plotted_cd)(self.image_ref, self.image_pro)
        else:
            self.print_message('Error ' + self.pacakage_name + ' does not have a module' + self.plotted_cd + \
                               '. Try again selecting the methods and packages first.')
        self.update_images()
        self.print_message('Current package: ' + self.package_name + '\n' + strin_of_values + \
                           'Current differnce map: ' + self.plotted_cd)
        self.print_message('Running time: ' + MU.format_time(time.clock() - self.start_time))

    def on_id_measures_click(self, button):
        popwin = MW.popup_window_with_list(self.list_of_selected_methods, sel_method=Gtk.SelectionMode.SINGLE,
                                           message="Measure")
        self.plotted_cd = popwin.list_items
        if not self.plotted_cd:
            self.return_to_default()
        if hasattr(self.selected_pacakage, self.plotted_cd):
            self.CD, _ = getattr(self.selected_pacakage, self.plotted_cd)(self.image_ref, self.image_pro)
        self.update_images()
        str_list_methods = ''
        str_list_methods += 'Reference ' + self.multimedia_file_ref.split('/')[-1] + '\n'
        str_list_methods += 'Sample ' + self.multimedia_file_pro.split('/')[-1] + '\n'
        for jj in self.list_of_selected_methods:
            if self.cd.has_key(self.multimedia_file_ref.split('/')[-1]):
                if self.cd[self.multimedia_file_ref.split('/')[-1]].has_key(self.multimedia_file_pro.split('/')[-1]):
                    if self.cd[self.multimedia_file_ref.split('/')[-1]][self.multimedia_file_pro.split('/')[-1]].has_key(jj):
                        str_list_methods += jj + ' = ' + "%.5f" % self.cd[self.multimedia_file_ref.split('/')[-1]] \
                            [self.multimedia_file_pro.split('/')[-1]][jj] + '\n'
            else:
                str_list_methods += 'Reference ' + self.multimedia_file_ref.split('/')[-1] + ', ' + jj + '\n'
                str_list_methods += 'Sample ' + self.multimedia_file_pro.split('/')[-1] + ', ' + jj + '\n'
        self.print_message('Current package: ' + self.package_name + '\n' + 'Selected methods:\n' + \
                           str_list_methods + 'Current differnce map: ' + self.plotted_cd)

    def single_sample_analysis(self, button):
        self.on_fid_groups_click()
        self.on_fid_measures_click()
        self.multimedia_file_ref = MW.load_file(self.window, 'img', 'Select your Reference Image')
        self.image_ref = ndimage.imread(self.multimedia_file_ref)
        self.multimedia_file_pro = MW.load_file(self.window, 'img', 'Select your Processed Image')
        self.image_pro = ndimage.imread(self.multimedia_file_pro)
        self.on_clicked()
        self.print_message("Process Finished!")

    def on_fid_groups_click(self, button=None):
        popwin = MW.popup_window_with_list(self.list_of_available_fidelity_groups, sel_method=Gtk.SelectionMode.SINGLE)
        self.package_name = popwin.list_items
        if not self.package_name:
            self.return_to_default()
        else:
            self.selected_pacakage = self.available_packages[self.package_name]
            self.print_message('Please select the new set of fidelity measures')
        str_list_methods = ''
        str_list_methods += 'Reference ' + self.multimedia_file_ref.split('/')[-1] + '\n'
        str_list_methods += 'Sample ' + self.multimedia_file_pro.split('/')[-1] + '\n'
        for jj in self.list_of_selected_methods:
            if self.cd.has_key(self.multimedia_file_ref.split('/')[-1]):
                if self.cd[self.multimedia_file_ref.split('/')[-1]].has_key(self.multimedia_file_pro.split('/')[-1]):
                    if self.cd[self.multimedia_file_ref.split('/')[-1]][self.multimedia_file_pro.split('/')[-1]].has_key(jj):
                        str_list_methods += jj + ' = ' + "%.5f" % self.cd[self.multimedia_file_ref.split('/')[-1]] \
                            [self.multimedia_file_pro.split('/')[-1]][jj] + '\n'
            else:
                str_list_methods += 'Sample ' + self.multimedia_file_ref.split('/')[-1] + ', ' + jj + '\n'
                str_list_methods += 'Sample ' + self.multimedia_file_pro.split('/')[-1] + ', ' + jj + '\n'
        self.print_message('Current package: ' + self.package_name + '\n' + 'Selected methods:\n' + \
                           str_list_methods + 'Current differnce map: ' + self.plotted_cd)

    def print_message(self, message):
        current_time = MW.get_time()
        self.textbuffer.insert_with_tags(self.textbuffer.get_end_iter(), '\n' + '\n' + current_time + '\n', self.h_tag)
        self.textbuffer.insert(self.textbuffer.get_end_iter(), message + '\n')

    def update_screen_size(self):
        screen = self.window.get_screen()
        monitors = []
        for m in range(screen.get_n_monitors()):
            monitors.append(screen.get_monitor_geometry(m))
        curmon = screen.get_monitor_at_window(screen.get_active_window())
        monitor_par = monitors[curmon]
        self.videoheight_default = (monitor_par.height - 130) / 2
        self.videowidth_default = (monitor_par.width - 300) / 2
        self.window.move(monitor_par.x, monitor_par.y)
        self.canvas.set_size_request(5. * self.videowidth_default / 8, self.videoheight_default)
        self.scrolledwindow.set_property("width-request", 3. * self.videowidth_default / 8)
        self.scrolledwindow.set_property("height-request", self.videoheight_default)
        self.status_bar.set_property("width-request", self.videowidth_default)

    def on_ti_measures_click(self, button):
        pr_file = self.multimedia_file_ref
        popwin = MW.popup_window_with_list(self.list_of_ref_samples, sel_method=Gtk.SelectionMode.SINGLE, \
                                           split_=True, message="Reference image")
        self.multimedia_file_ref = popwin.list_items
        if not self.multimedia_file_ref:
            self.multimedia_file_ref = pr_file
            self.print_message("Error in processed file selection. Returning to sample: " + pr_file.split('/')[-1])
        else:
            flag = False
            for ii in self.list_of_ref_samples:
                if self.multimedia_file_ref in ii:
                    nw_file = ii
                    flag = True
            if not flag:
                self.multimedia_file_ref = pr_file
                self.print_message("Error finding selection. Returning to sample: " + pr_file.split('/')[-1])
            else:
                self.multimedia_file_ref = nw_file
        self.image_ref = ndimage.imread(self.multimedia_file_ref)
        pr_file = self.multimedia_file_pro
        popwin = MW.popup_window_with_list(self.list_of_test_samples[self.multimedia_file_ref.split('/')[-1]], \
                                           sel_method=Gtk.SelectionMode.SINGLE, split_=True, message="Test image")
        self.multimedia_file_pro = popwin.list_items
        if not self.multimedia_file_pro:
            self.multimedia_file_pro = pr_file
            self.print_message("Error in processed file selection. Returning to sample: " + pr_file.split('/')[-1])
        else:
            flag = False
            for ii in self.list_of_test_samples[self.multimedia_file_ref.split('/')[-1]]:
                if self.multimedia_file_pro in ii:
                    nw_file = ii
                    flag = True
            if not flag:
                self.multimedia_file_pro = pr_file
                self.print_message("Error finding selection. Returning to sample: " + pr_file.split('/')[-1])
            else:
                self.multimedia_file_pro = nw_file
        self.image_pro = ndimage.imread(self.multimedia_file_pro)
        if not hasattr(self.selected_pacakage, self.plotted_cd):
            self.return_to_default()
        self.CD, _ = getattr(self.selected_pacakage, self.plotted_cd)(self.image_ref, self.image_pro)
        self.update_images()
        self.plot_current_selection()
        str_list_methods = ''
        str_list_methods += 'Reference ' + self.multimedia_file_ref.split('/')[-1] + '\n'
        str_list_methods += 'Sample ' + self.multimedia_file_pro.split('/')[-1] + '\n'
        for jj in self.list_of_selected_methods:
            if self.cd.has_key(self.multimedia_file_ref.split('/')[-1]):
                if self.cd[self.multimedia_file_ref.split('/')[-1]].has_key(self.multimedia_file_pro.split('/')[-1]):
                    if self.cd[self.multimedia_file_ref.split('/')[-1]][self.multimedia_file_pro.split('/')[-1]].has_key(jj):
                        str_list_methods += jj + ' = ' + "%.5f" % self.cd[self.multimedia_file_ref.split('/')[-1]] \
                            [self.multimedia_file_pro.split('/')[-1]][jj] + '\n'
            else:
                str_list_methods += 'Reference ' + self.multimedia_file_ref.split('/')[-1] + ', ' + jj + '\n'
                str_list_methods += 'Sample ' + self.multimedia_file_pro.split('/')[-1] + ', ' + jj + '\n'
        self.print_message('Current package: ' + self.package_name + '\n' + 'Selected methods:\n' + \
                           str_list_methods + 'Current differnce map: ' + self.plotted_cd)

    def on_click_single_multiple(self, button):
        self.on_fid_groups_click()
        self.on_fid_measures_click()
        self.multimedia_file_ref = MW.load_file(self.window, 'img', 'Select your Reference Image')
        self.list_of_ref_samples = [self.multimedia_file_ref]
        self.image_ref = ndimage.imread(self.multimedia_file_ref)
        self.list_of_test_samples = {}
        self.list_of_test_samples[self.multimedia_file_ref.split('/')[-1]] = \
            MW.load_file(self.window, 'img', 'Select your Processed Images', multiple=True)
        if self.list_of_test_samples[self.multimedia_file_ref.split('/')[-1]]:
            for ii in self.list_of_test_samples[self.multimedia_file_ref.split('/')[-1]]:
                self.multimedia_file_pro = ii
                self.image_pro = ndimage.imread(self.multimedia_file_pro)
                self.on_clicked()
                while Gtk.events_pending():
                    Gtk.main_iteration()
                if self.stopped:
                    return False
            self.x_axis = self.plotted_cd
            self.y_axis = self.plotted_cd
        else:
            self.print_message("Error in sample selection. Please select again your samples.")
        self.print_message("Process Finished!")

    def on_mx_my_plot_click(self, button):
        popwin = MW.popup_window_with_list(self.list_of_selected_methods, sel_method=Gtk.SelectionMode.SINGLE,
                                           message="X axis measure")
        self.x_axis = popwin.list_items
        if not self.x_axis:
            self.x_axis = 'cd00_deltaE2000'
        popwin = MW.popup_window_with_list(self.list_of_selected_methods, sel_method=Gtk.SelectionMode.SINGLE,
                                           message="Y axis measure")
        self.y_axis = popwin.list_items
        if not self.y_axis:
            self.y_axis = 'cd00_deltaE2000'
        self.print_message("Current x axis: " + self.x_axis)
        self.print_message("Current y axis: " + self.y_axis)
        self.plot_current_selection()

    def return_to_default(self):
        self.cd = {}
        self.x_axis = 'cd00_deltaE2000'
        self.y_axis = 'cd00_deltaE2000'
        self.multimedia_file_ref = self.working_path + '/sample_images/test_ref_0.bmp'
        self.image_ref = ndimage.imread(self.multimedia_file_ref)
        self.multimedia_file_pro = self.working_path + '/sample_images/test_pro_0.bmp'
        self.image_pro = ndimage.imread(self.multimedia_file_pro)
        self.list_of_test_samples = {}
        self.list_of_test_samples[self.multimedia_file_ref.split('/')[-1]] = [self.multimedia_file_pro]
        self.list_of_ref_samples = [self.multimedia_file_ref]
        self.cd[self.multimedia_file_ref.split('/')[-1]] = {}
        self.cd[self.multimedia_file_ref.split('/')[-1]][self.multimedia_file_pro.split('/')[-1]] = {}
        self.plotted_cd = 'cd00_deltaE2000'
        self.content_features = {}
        self.list_of_selected_methods = ['cd00_deltaE2000']
        self.list_of_methods = ['cd00_deltaE2000']
        self.package_name = 'cd_measures_pack'
        self.selected_pacakage = self.available_packages[self.package_name]
        self.print_message(
            'Error in one or more options.\n Returning to default parameters.\n Please Select your options again.')
        self.on_clicked()

    def on_add_mos_click(self, button=None):
        temp = MW.load_file(self.window, 'txt', 'Select your MOS or measure file')
        if temp:
            popwin = MW.popup_window_with_text_input('Name your measure: ')
            with  open(temp) as f:
                for line in f:
                    temp_string = line.split()
                    if self.cd.has_key(temp_string[0]):
                        if self.cd[temp_string[0]].has_key(temp_string[1]):
                            self.cd[temp_string[0]][temp_string[1]][popwin.file_name] = float(temp_string[2])
            self.list_of_selected_methods.append(popwin.file_name)
            self.print_message("File " + temp + " loaded.")
        else:
            self.print_message("Error Selecting dmos/measure File. Please Try again")

    def on_add_nfg_click(self, button=None):
        popwin = MW.popup_window_with_text_input()
        self.list_of_available_fidelity_groups.append(popwin.file_name)  # IT is a string with the name of the packages
        file('./list_of_packages', 'w').write("\n".join(self.list_of_available_fidelity_groups) + "\n")
        self.get_list_of_packages("./list_of_packages")
        self.print_message('New fidelity pack add to the default packages.')

    def get_list_of_packages(self, file_name):
        list_of_packages = []
        list_of_modules = {}
        with  open(file_name) as f:
            for line in f:
                line = line.replace('\n', '')
                if self.verify_package(line):
                    list_of_packages.append(line)
                    list_of_modules[line] = importlib.import_module(line)
                else:
                    try:
                        self.print_message('Verify your list_of_packages file. One ore more lines' + \
                                           'could be corrupted or empty :' + line)
                    except AttributeError:
                        print 'Verify your list_of_packages file. One ore more lines' + \
                              'could be corrupted or empty :' + line
        self.available_packages = list_of_modules  # IT is actually the set of modules to be call
        self.list_of_available_fidelity_groups = list_of_packages  # IT is a string with the name of the packages

    def verify_package(self, package_name):
        try:
            result = importlib.import_module(package_name)
            if result:
                return result
        except ImportError:
            return None

    def on_load_click(self, button=None):
        temp = MW.load_file(self.window, 'txt', 'Select your .iFAS file')
        with open(temp) as f:
            self.cd, self.multimedia_file_ref, self.multimedia_file_pro, self.list_of_test_samples, \
            self.list_of_ref_samples, self.plotted_cd, self.list_of_selected_methods, \
            self.list_of_methods, self.package_name, self.x_axis, self.y_axis = pickle.load(f)
        self.selected_pacakage = self.available_packages[self.package_name]
        self.image_ref = ndimage.imread(self.multimedia_file_ref)
        self.image_pro = ndimage.imread(self.multimedia_file_pro)
        self.selected_pacakage = self.available_packages[self.package_name]
        if hasattr(self.selected_pacakage, self.plotted_cd):
            self.CD, _ = getattr(self.selected_pacakage, self.plotted_cd)(self.image_ref, self.image_pro)
        else:
            self.print_message('Error ' + self.pacakage_name + ' does not have a module' + self.plotted_cd + \
                               '. Try again selecting the methods and packages first.')
        self.update_images()
        str_list_methods = ''
        str_list_methods += 'Reference ' + self.multimedia_file_ref.split('/')[-1] + '\n'
        str_list_methods += 'Sample ' + self.multimedia_file_pro.split('/')[-1] + '\n'
        for jj in self.list_of_selected_methods:
            if self.cd.has_key(self.multimedia_file_ref.split('/')[-1]):
                if self.cd[self.multimedia_file_ref.split('/')[-1]].has_key(self.multimedia_file_pro.split('/')[-1]):
                    if self.cd[self.multimedia_file_ref.split('/')[-1]] \
                            [self.multimedia_file_pro.split('/')[-1]].has_key(jj):
                        str_list_methods += jj + ' = ' + "%.5f" % self.cd[self.multimedia_file_ref.split('/')[-1]] \
                            [self.multimedia_file_pro.split('/')[-1]][jj] + '\n'
            else:
                str_list_methods += 'Reference ' + self.multimedia_file_ref.split('/')[-1] + ', ' + jj + '\n'
                str_list_methods += 'Sample ' + self.multimedia_file_pro.split('/')[-1] + ', ' + jj + '\n'
        self.print_message('Current package: ' + self.package_name + '\n' + 'Selected methods:\n' + \
                           str_list_methods + 'Current differnce map: ' + self.plotted_cd)
        self.plot_current_selection()
        self.print_message("Load completed")

    def on_save_click(self, button=None):
        popwin = MW.popup_window_with_text_input('Name your file: ')
        with open(self.working_path + '/' + popwin.file_name + '.iFAS', 'w') as f:
            pickle.dump([self.cd, self.multimedia_file_ref, self.multimedia_file_pro, self.list_of_test_samples, \
                         self.list_of_ref_samples, self.plotted_cd, self.list_of_selected_methods, \
                         self.list_of_methods, self.package_name, self.x_axis, self.y_axis], f)
        self.print_message("Save completed")

    def on_click_multiple_source(self, button=None):
        self.start_time = time.clock()
        self.on_fid_groups_click()
        self.on_fid_measures_click()
        self.list_of_test_samples = {}
        self.list_of_ref_samples = []
        temp = MW.load_file(self.window, 'txt', 'Select your .iFASpro file')
        file_location = '/'.join(temp.split('/')[0:-1])
        if temp:
            with  open(temp) as f:
                for line in f:
                    temp_string = line.split()
                    if not temp_string[0] == '\n':
                        self.multimedia_file_ref = file_location + '/' + temp_string[0]
                        self.list_of_ref_samples.append(self.multimedia_file_ref)
                        self.image_ref = ndimage.imread(self.multimedia_file_ref)
                        temp_string = temp_string[1:]
                        if temp_string:
                            self.list_of_test_samples[self.multimedia_file_ref.split('/')[-1]] = []
                            for ii in temp_string:
                                while Gtk.events_pending():
                                    Gtk.main_iteration()
                                if self.stopped:
                                    return False
                                if not ii == '\n':
                                    self.multimedia_file_pro = file_location + '/' + ii
                                    self.list_of_test_samples[self.multimedia_file_ref.split('/')[-1]]. \
                                        append(self.multimedia_file_pro)
                                    self.image_pro = ndimage.imread(self.multimedia_file_pro)
                                    self.on_clicked()
                        else:
                            while Gtk.events_pending():
                                Gtk.main_iteration()
                            self.print_message("Error in samples selection. Please select again your samples.")
            self.print_message("Process Finished!")
            self.x_axis = self.plotted_cd
            self.y_axis = self.plotted_cd
        else:
            self.print_message("Error Selecting iFASpro File. Please Try again")

    def plot_current_selection(self):
        vec = np.zeros((len(self.cd[self.multimedia_file_ref.split('/')[-1]]), 2))
        count = 0
        if self.cd.has_key(self.multimedia_file_ref.split('/')[-1]):
            for ii in self.list_of_test_samples[self.multimedia_file_ref.split('/')[-1]]:
                if self.cd[self.multimedia_file_ref.split('/')[-1]].has_key(ii.split('/')[-1]):
                    if self.cd[self.multimedia_file_ref.split('/')[-1]][ii.split('/')[-1]].has_key(self.x_axis) and \
                            self.cd[self.multimedia_file_ref.split('/')[-1]][ii.split('/')[-1]].has_key(
                                self.y_axis):
                        vec[count, :] = np.array(
                            [self.cd[self.multimedia_file_ref.split('/')[-1]][ii.split('/')[-1]] \
                                 [self.x_axis], self.cd[self.multimedia_file_ref.split('/')[-1]] \
                                 [ii.split('/')[-1]][self.y_axis]])
                        count += 1
        if count > 0:
            self.plotted.set_data(vec[:, 0], vec[:, 1])
            self.axis.set_ylim(
                [np.min(vec[:, 1]) - 0.05 * np.min(vec[:, 1]), np.max(vec[:, 1]) + 0.05 * np.max(vec[:, 1])])
            self.axis.set_xlim(
                [np.min(vec[:, 0]) - 0.05 * np.min(vec[:, 0]), np.max(vec[:, 0]) + 0.05 * np.max(vec[:, 0])])
            self.axis.relim()
            self.axis.autoscale_view(True, True, True)
            self.axis.set_ylabel(self.y_axis, fontsize=12)
            self.axis.set_xlabel(self.x_axis, fontsize=12)
            self.figure.canvas.draw()
            p, s, t, pd = MU.compute_1dcorrelatiosn(vec[:, 0], vec[:, 1])
            self.print_message("Pearsonr = " + "%.5f" % p + "\n" + "Spearmanr = " + "%.5f" % s + \
                               "\n" + "Kendalltau = " + "%.5f" % t + "\n" + "Correlation distance = " + "%.5f" % pd)

    def on_barplot_click(self, button=None):
        corrs = {}
        for ii in self.list_of_selected_methods:
            if not ii == 'dmos':
                vec = []
                for jj in self.list_of_ref_samples:
                    if self.cd.has_key(jj.split('/')[-1]):
                        for kk in self.list_of_test_samples[jj.split('/')[-1]]:
                            if self.cd[jj.split('/')[-1]].has_key(kk.split('/')[-1]):
                                if self.cd[jj.split('/')[-1]][kk.split('/')[-1]].has_key(ii):
                                    if self.cd[jj.split('/')[-1]][kk.split('/')[-1]].has_key('dmos'):
                                        vec.append([self.cd[jj.split('/')[-1]][kk.split('/')[-1]][ii], \
                                                    self.cd[jj.split('/')[-1]][kk.split('/')[-1]]['dmos']])
                vec = np.asarray(vec)
                p, s, t, pd = MU.compute_1dcorrelatiosn(vec[:, 0], vec[:, 1])
                corrs[ii] = np.abs(np.asarray([p, s, t, pd]))
                self.print_message(ii + '\n' + 'Pearsonr: ' + "%.5f" % p + '\n' + 'Spearmanr: ' + "%.5f" % s \
                                   + '\n' + 'Kendalltau: ' + "%.5f" % t + '\n' + 'Correlation distance: ' + "%.5f" % pd)
        pmax = 0.
        smax = 0.
        tmax = 0.
        pdmax = 0.
        for ii in corrs:
            if np.abs(corrs[ii][0]) > pmax:
                pmax = corrs[ii][0]
                pos_p = ii
            if np.abs(corrs[ii][1]) > smax:
                smax = corrs[ii][1]
                pos_s = ii
            if np.abs(corrs[ii][2]) > tmax:
                tmax = corrs[ii][2]
                pos_t = ii
            if np.abs(corrs[ii][3]) > pdmax:
                pdmax = corrs[ii][3]
                pos_pd = ii
        message = "Best according to Pearsonr is " + pos_p + ': ' + \
                  "%.5f" % pmax
        message += "\nBest according to Spearmanr is " + pos_s + ': ' + \
                   "%.5f" % smax
        message += "\nBest according to Kendalltau is " + pos_t + ': ' + \
                   "%.5f" % tmax
        message += "\nBest according to Correlation distance is " + pos_pd + \
                   ': ' + "%.5f" % pdmax
        self.print_message(message)
        temp = MW.popup_window_with_bar_plot(self, corrs)

    def on_boxplot_click(self, button=None):
        corrs = {}
        for ii in self.list_of_selected_methods:
            if not ii == 'dmos':
                corrs[ii] = []
                for jj in self.list_of_ref_samples:
                    if self.cd.has_key(jj.split('/')[-1]):
                        vec = []
                        for kk in self.list_of_test_samples[jj.split('/')[-1]]:
                            if self.cd[jj.split('/')[-1]].has_key(kk.split('/')[-1]):
                                if self.cd[jj.split('/')[-1]][kk.split('/')[-1]].has_key(ii):
                                    if self.cd[jj.split('/')[-1]][kk.split('/')[-1]].has_key('dmos'):
                                        vec.append([self.cd[jj.split('/')[-1]][kk.split('/')[-1]][ii], \
                                                    self.cd[jj.split('/')[-1]][kk.split('/')[-1]]['dmos']])
                        vec = np.asarray(vec)
                        p, s, t, pd = MU.compute_1dcorrelatiosn(vec[:, 0], vec[:, 1])
                        corrs[ii].append([p, s, t, pd])
                        self.print_message(ii + '\n' + jj.split('/')[-1] + '\n' + 'Pearsonr: ' + "%.5f" % p + '\n' + \
                                           'Spearmanr: ' + "%.5f" % s + '\n' + 'Kendalltau: ' + "%.5f" % t + '\n' + \
                                           'Correlation distance: ' + "%.5f" % pd)
                corrs[ii] = np.asarray(corrs[ii])
        temp = MW.popup_window_with_box_plot(self, corrs)

    def on_scatterplot_click(self, button=None):
        values = {}
        popwin = MW.popup_window_with_list(self.list_of_selected_methods, sel_method=Gtk.SelectionMode.SINGLE,
                                           message="X axis measure")
        x_axis = popwin.list_items
        popwin = MW.popup_window_with_list(self.list_of_selected_methods, sel_method=Gtk.SelectionMode.SINGLE,
                                           message="Y axis measure")
        y_axis = popwin.list_items
        for jj in self.list_of_ref_samples:
            if self.cd.has_key(jj.split('/')[-1]):
                values[jj.split('/')[-1]] = []
                for kk in self.list_of_test_samples[jj.split('/')[-1]]:
                    if self.cd[jj.split('/')[-1]].has_key(kk.split('/')[-1]):
                        if self.cd[jj.split('/')[-1]][kk.split('/')[-1]].has_key(x_axis) and \
                                self.cd[jj.split('/')[-1]][kk.split('/')[-1]].has_key(y_axis):
                            values[jj.split('/')[-1]].append([self.cd[jj.split('/')[-1]][kk.split('/')[-1]][x_axis], \
                                                              self.cd[jj.split('/')[-1]][kk.split('/')[-1]][y_axis]])
                values[jj.split('/')[-1]] = np.asarray(values[jj.split('/')[-1]])
        temp = MW.popup_window_with_scatterplot(self, values, (x_axis, y_axis))

    def on_regression_click(self, button=None, data=None):
        popwin = MW.popup_window_with_list(self.list_of_ref_samples, sel_method=Gtk.SelectionMode.MULTIPLE, \
                                           split_=True, message="Reference image TRAINING")
        list_of_training = popwin.list_items
        popwin = MW.popup_window_with_list(self.list_of_ref_samples, sel_method=Gtk.SelectionMode.MULTIPLE, \
                                           split_=True, message="Reference image TEST")
        list_of_test = popwin.list_items
        popwin = MW.popup_window_with_list(self.list_of_selected_methods, sel_method=Gtk.SelectionMode.SINGLE,
                                           message="Independent variable")
        x_axis = popwin.list_items
        popwin = MW.popup_window_with_list(self.list_of_selected_methods, sel_method=Gtk.SelectionMode.SINGLE,
                                           message="Dependent variable")
        y_axis = popwin.list_items
        x = []
        y = []
        for ii in list_of_training:
            if self.cd.has_key(ii):
                for jj in self.list_of_test_samples[ii]:
                    if self.cd[ii].has_key(jj.split('/')[-1]):
                        if self.cd[ii][jj.split('/')[-1]].has_key(x_axis) and self.cd[ii][jj.split('/')[-1]].has_key(
                                y_axis):
                            x.append(self.cd[ii][jj.split('/')[-1]][x_axis])
                            y.append(self.cd[ii][jj.split('/')[-1]][y_axis])
        x = np.array(x)
        y = np.array(y)
        aopt = OT.optimize_function(x, y, fun_type=data)
        y_est = OT.gen_data(x, aopt, fun_type=data)
        messagea = ''
        for ii in range(len(aopt)):
            messagea += 'a' + str(ii) + ": %.5f" % aopt[ii] + '\n'
        self.print_message("The optimal parameters for function\n" + OT.fun_text(data) + "\n" + messagea)
        p, s, t, pd = MU.compute_1dcorrelatiosn(y, y_est)
        self.print_message('Training values: ' + '\n' + 'Pearsonr: ' + "%.5f" % p + '\n' + \
                           'Spearmanr: ' + "%.5f" % s + '\n' + 'Kendalltau: ' + "%.5f" % t + '\n' + \
                           'Correlation distance: ' + "%.5f" % pd)
        x_test = []
        y_test = []
        for ii in list_of_test:
            if self.cd.has_key(ii):
                for jj in self.list_of_test_samples[ii]:
                    if self.cd[ii].has_key(jj.split('/')[-1]):
                        if self.cd[ii][jj.split('/')[-1]].has_key(x_axis) and self.cd[ii][jj.split('/')[-1]].has_key(
                                y_axis):
                            x_test.append(self.cd[ii][jj.split('/')[-1]][x_axis])
                            y_test.append(self.cd[ii][jj.split('/')[-1]][y_axis])
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        y_est_test = OT.gen_data(x_test, aopt, fun_type=data)
        p, s, t, pd = MU.compute_1dcorrelatiosn(y_test, y_est_test)
        self.print_message('Testing values: ' + '\n' + 'Pearsonr: ' + "%.5f" % p + '\n' + \
                           'Spearmanr: ' + "%.5f" % s + '\n' + 'Kendalltau: ' + "%.5f" % t + '\n' + \
                           'Correlation distance: ' + "%.5f" % pd)
        data_to_plot = {}
        xfull = np.linspace(np.minimum(np.min(x), np.min(x_test)), np.maximum(np.max(x), np.max(x_test)), 100)
        data_to_plot['rl'] = np.transpose(np.vstack((xfull, OT.gen_data(xfull, aopt, fun_type=data))))
        data_to_plot['t'] = np.transpose(np.vstack((x, y)))
        data_to_plot['s'] = np.transpose(np.vstack((x_test, y_test)))
        temp = MW.popup_window_with_scatterplot_regression(self, data_to_plot, (x_axis, y_axis))

    def on_boxplot_diff_click(self, button=None):
        popwin = MW.popup_window_with_list(self.list_of_ref_samples, sel_method=Gtk.SelectionMode.MULTIPLE, \
                                           split_=True, message="Reference images for the analysis")
        list_of_ref = popwin.list_items
        popwin = MW.popup_window_with_list(self.list_of_selected_methods, sel_method=Gtk.SelectionMode.MULTIPLE,
                                           message="Select algorithms for the analysis")
        list_of_methods = popwin.list_items
        popwin = MW.popup_window_with_box_plot_differences(Object=self, data=self.cd, ref_names=list_of_ref, \
                                                           list_methods=list_of_methods)

    def on_hist_content_features(self, button=None):
        list_content_features = inspect.getmembers(CF, inspect.isfunction)
        list_of_methods = []
        for ii in range(len(list_content_features)):
            list_of_methods.append(list_content_features[ii][0])
        popwin = MW.popup_window_with_list(list_of_methods, sel_method=Gtk.SelectionMode.SINGLE,
                                           message="Content features")
        vec = []
        vec_names = []
        for ii in self.list_of_ref_samples:
            self.content_features[ii.split('/')[-1]] = getattr(CF, popwin.list_items)(ndimage.imread(ii))
            vec.append(self.content_features[ii.split('/')[-1]])
            vec_names.append(ii.split('/')[-1])
        popwin = MW.popup_window_with_content_hist(Object=self, data=vec)
        print vec

    def on_heatmap_click(self, button=None):
        self.print_message("Heating the map")
        values = []
        popwin = MW.popup_window_with_list(self.list_of_selected_methods, sel_method=Gtk.SelectionMode.SINGLE,
                                           message="X axis measure")
        x_axis = popwin.list_items
        popwin = MW.popup_window_with_list(self.list_of_selected_methods, sel_method=Gtk.SelectionMode.SINGLE,
                                           message="Y axis measure")
        y_axis = popwin.list_items
        for jj in self.list_of_ref_samples:
            if self.cd.has_key(jj.split('/')[-1]):
                for kk in self.list_of_test_samples[jj.split('/')[-1]]:
                    if self.cd[jj.split('/')[-1]].has_key(kk.split('/')[-1]):
                        if self.cd[jj.split('/')[-1]][kk.split('/')[-1]].has_key(x_axis) and \
                                self.cd[jj.split('/')[-1]][kk.split('/')[-1]].has_key(y_axis) and \
                                self.cd[jj.split('/')[-1]][kk.split('/')[-1]].has_key('dmos'):
                            values.append([self.cd[jj.split('/')[-1]][kk.split('/')[-1]][x_axis],\
                                           self.cd[jj.split('/')[-1]][kk.split('/')[-1]][y_axis],\
                                           self.cd[jj.split('/')[-1]][kk.split('/')[-1]]['dmos']])
        values = np.asarray(values)
        data = {x_axis: values[:, 0], y_axis: values[:, 1], 'dmos': values[:, 2]}
        temp = MW.popup_window_with_parametermap(self, data, para_rangex=(0.0, 1), para_rangey=(0.0, 1), name_axis=(x_axis, y_axis))

    def on_multiple_plot(self, button=None, data=None):
        prelist = []
        for ii in sorted(self.list_of_test_samples):
            for jj in sorted(self.list_of_test_samples[ii]):
                prelist.append(jj)
        popwin = MW.popup_window_with_list(sorted(prelist, key=lambda s: s[-9:-1]), sel_method=Gtk.SelectionMode.MULTIPLE,\
                                           split_=True, message="First distortion")
        list_of_first_dis = popwin.list_items
        popwin = MW.popup_window_with_list(sorted(prelist, key=lambda s: s[-9:-1]), sel_method=Gtk.SelectionMode.MULTIPLE,\
                                           split_=True, message="Second distortion")
        list_of_second_dis = popwin.list_items
        popwin = MW.popup_window_with_list(sorted(prelist, key=lambda s: s[-9:-1]), sel_method=Gtk.SelectionMode.MULTIPLE,\
                                           split_=True, message="Third distortion")
        list_of_third_dis = popwin.list_items
        popwin = MW.popup_window_with_list(self.list_of_selected_methods, sel_method=Gtk.SelectionMode.SINGLE,
                                           message="X axis")
        x_axis = popwin.list_items
        popwin = MW.popup_window_with_list(self.list_of_selected_methods, sel_method=Gtk.SelectionMode.SINGLE,
                                           message="Y axis")
        y_axis = popwin.list_items
        vals = []
        for jj in self.list_of_ref_samples:
            if self.cd.has_key(jj.split('/')[-1]):
                for kk in list_of_first_dis:
                    if self.cd[jj.split('/')[-1]].has_key(kk):
                        if self.cd[jj.split('/')[-1]][kk].has_key(x_axis) and self.cd[jj.split('/')[-1]][kk].has_key(y_axis):
                            vals.append([self.cd[jj.split('/')[-1]][kk][x_axis], self.cd[jj.split('/')[-1]][kk][y_axis], 0])
                for kk in list_of_second_dis:
                    if self.cd[jj.split('/')[-1]].has_key(kk):
                        if self.cd[jj.split('/')[-1]][kk].has_key(x_axis) and self.cd[jj.split('/')[-1]][kk].has_key(y_axis):
                            vals.append([self.cd[jj.split('/')[-1]][kk][x_axis], self.cd[jj.split('/')[-1]][kk][y_axis], 1])
                for kk in list_of_third_dis:
                    if self.cd[jj.split('/')[-1]].has_key(kk):
                        if self.cd[jj.split('/')[-1]][kk].has_key(x_axis) and self.cd[jj.split('/')[-1]][kk].has_key(y_axis):
                            vals.append([self.cd[jj.split('/')[-1]][kk][x_axis], self.cd[jj.split('/')[-1]][kk][y_axis], 2])
        temp = MW.popup_window_with_scatterplot_wizard(self, np.array(vals), (x_axis, y_axis))

    def on_guide_click(self, button=None):
        self.print_message("Do you need some help?\nPlease see our help GUIDE in pdf format")
        self.on_about_click()
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, self.working_path + '/documents/Help.pdf'])

    def on_about_click(self, button=None):
        self.print_message("iFAS alpha-Version Edition " + 'December 2016' + '\n' + \
                           'Copyright 2016 Benhur Ortiz Jaramillo\n' + \
                           'This program comes with absolutely no warranty.\n' + \
                           'See the GNU General Public License for details.')


# ~ Uncomment the following three lines for standalone running
if __name__ == "__main__":
    Main()
    Gtk.main()
