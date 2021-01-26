import gi

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
from gi.repository import Pango
from time import gmtime, strftime
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas
from matplotlib.backends.backend_gtk3 import NavigationToolbar2GTK3 as NavigationToolbar
import numpy as np
import myUtilities
import logging
import os


class iFasLog(object):
    def __init__(self, pathLog):
        currentTime = getTime(shortFormat=True)
        try:
            os.makedirs(pathLog + '/log')
        except OSError:
            pass
        logging.basicConfig(filename=pathLog + '/log/log_' + currentTime + '.log',
                            format='%(asctime)s %(message)s', level=logging.INFO)
        logging.info('Starting logging for current session')

    def onLogging(self, logType='info', message=''):
        logging.info(logType + ': ' + message)
        # if logType == 'info':
        #     logging.info(logType + ': ' + message)
        # elif logType == 'warning':
        #     logging.warning(logType + ': ' + message)
        # elif logType == 'error':
        #     logging.error(logType + ': ' + message)
        # else:
        #     logging.debug(logType + ': ' + message)


def load_file(MainWindow, type, message, multiple=False):
    if type == 'img':
        filter = Gtk.FileFilter()
        filter.set_name("Image")
        filter.add_mime_type("image/png")
        filter.add_mime_type("image/jpeg")
        filter.add_mime_type("image/gif")
        filter.add_pattern("*.png")
        filter.add_pattern("*.jpg")
        filter.add_pattern("*.gif")
        filter.add_pattern("*.tif")
        filter.add_pattern("*.bmp")
        filter.add_pattern("*.xpm")
    elif type == 'avi':
        filter = Gtk.FileFilter()
        filter.set_name("Video")
        filter.add_mime_type("video/avi")
        filter.add_pattern("*.avi")
    elif type == 'txt':
        filter = Gtk.FileFilter()
        filter.set_name("Text")
        filter.add_mime_type("text/plain")
        filter.add_pattern("*.txt")
        filter.add_mime_type("text/plain")
        filter.add_pattern("*.")
        filter.add_mime_type("text/plain")
        filter.add_pattern("*.iFAS")
        filter.add_mime_type("text/plain")
        filter.add_pattern("*.iFASpro")
    chooser = Gtk.FileChooserDialog(message, MainWindow, Gtk.FileChooserAction.OPEN,
                                    (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OPEN, Gtk.ResponseType.OK))
    chooser.add_filter(filter)
    chooser.set_select_multiple(multiple)
    if chooser.run() == Gtk.ResponseType.OK:
        if multiple:
            temp = chooser.get_filenames()
        else:
            temp = chooser.get_filename()
    else:
        chooser.destroy()
        return None
    chooser.destroy()
    return temp


def getTime(shortFormat=False):
    if shortFormat:
        return strftime("%Y_%m_%d_%H_%M_%S", gmtime())
    else:
        return strftime("%Y-%m-%d %H:%M:%S", gmtime())


class popupWindowWithList(object):
    def on_changed(self, selection):
        (model, pathlist) = selection.get_selected_rows()
        self.list_items = []
        for ii in range(0, len(pathlist)):
            self.list_items.append(model[pathlist[ii]][0])
        return True

    def on_changed_single(self, selection):
        (model, iter) = selection.get_selected()
        self.list_items = model[iter][0]
        return True

    def on_click_me_clicked(self, button=None, widget=None):
        self.window.destroy()
        Gtk.main_quit()
        return True

    def __init__(self, list_to_show, sel_method=Gtk.SelectionMode.SINGLE, split_=False, message="Package"):
        self.list_items = []
        self.window = Gtk.Window()
        self.window.set_title("iFAS - " + message)
        self.window.connect("delete-event", self.on_click_me_clicked)
        self.window.connect("destroy", self.on_click_me_clicked)
        listmodel = Gtk.ListStore(str)
        for ii in list_to_show:
            if split_:
                listmodel.append([ii.split('/')[-1]])
            else:
                listmodel.append([ii])
        view = Gtk.TreeView(model=listmodel)
        cell = Gtk.CellRendererText()
        cell.props.weight_set = True
        cell.props.weight = Pango.Weight.BOLD
        cell.set_property('size-points', 15)
        col = Gtk.TreeViewColumn(message + ' Name', cell, text=0)
        view.append_column(col)
        button = Gtk.Button.new_with_label("Finish Selection")
        if sel_method is Gtk.SelectionMode.MULTIPLE:
            view.get_selection().connect("changed", self.on_changed)
        else:
            view.get_selection().connect("changed", self.on_changed_single)
        button.connect("clicked", self.on_click_me_clicked)
        view.get_selection().set_mode(sel_method)  # Gtk.SelectionMode.MULTIPLE or Gtk.SelectionMode.SINGLE
        self.hbox = Gtk.Box()
        self.window.add(self.hbox)
        self.scrolled_window()
        self.scrolledwindow.add(view)
        self.hbox.pack_start(button, True, True, 0)
        self.window.show_all()
        Gtk.main()

    def scrolled_window(self):
        self.scrolledwindow = Gtk.ScrolledWindow()
        self.scrolledwindow.set_hexpand(True)
        self.scrolledwindow.set_vexpand(True)
        self.scrolledwindow.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.ALWAYS)
        self.scrolledwindow.set_property("width-request", 256)
        self.scrolledwindow.set_property("height-request", 512)
        self.hbox.pack_start(self.scrolledwindow, True, True, 0)


class popupWindowWithTextInput(object):
    def on_click_me_clicked(self, button=None, widget=None):
        self.file_name = self.entry.get_text()
        self.window.destroy()
        Gtk.main_quit()
        return True

    def __init__(self, message='Type the file name of your python file without extension .py: '):
        self.file_name = None
        self.window = Gtk.Window()
        self.window.set_title("iFAS - " + message)
        self.window.connect("delete-event", self.on_click_me_clicked)
        self.window.connect("destroy", self.on_click_me_clicked)
        button = Gtk.Button.new_with_label("Finish Selection")
        button.connect("clicked", self.on_click_me_clicked)
        button.modify_font(Pango.FontDescription('Sans 16'))
        self.hbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.window.add(self.hbox)
        self.entry = Gtk.Entry()
        self.hbox.pack_start(self.entry, True, True, 0)
        self.hbox.pack_start(button, True, True, 0)
        self.window.show_all()
        Gtk.main()


class popupWindowWithLabel(object):
    def on_click_me_clicked(self, button=None, widget=None):
        self.window.destroy()
        Gtk.main_quit()
        return True

    def __init__(self, message='Error check your log file!'):
        self.window = Gtk.Window()
        self.window.set_title("iFAS - " + message)
        self.window.connect("delete-event", self.on_click_me_clicked)
        self.window.connect("destroy", self.on_click_me_clicked)
        button = Gtk.Button.new_with_label("OK")
        button.connect("clicked", self.on_click_me_clicked)
        button.modify_font(Pango.FontDescription('Sans 16'))
        self.hbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.window.add(self.hbox)
        label = Gtk.Label(message)
        label.modify_font(Pango.FontDescription('Sans 18'))
        self.hbox.pack_start(label, True, True, 0)
        self.hbox.pack_start(button, True, True, 0)
        self.window.show_all()
        Gtk.main()


class popupWindowWithBarPlot(object):
    def on_click_me_clicked(self, button=None, widget=None):
        self.window.destroy()
        Gtk.main_quit()
        return True

    def __init__(self, Object=None, data={}):
        self.window = Gtk.Window()
        self.window.set_title("iFAS - Bar plot")
        self.window.set_default_size(800, 500)
        self.window.connect("delete-event", self.on_click_me_clicked)
        self.window.connect("destroy", self.on_click_me_clicked)
        f = Figure()
        acorrs = f.add_subplot(111)
        acorrs.tick_params(labelsize=16)
        p = []
        s = []
        # t = []
        pd = []
        labels = []
        for ii in sorted(data):
            labels.append(ii.split('_')[0])
            p.append(data[ii][0])
            s.append(data[ii][1])
            # t.append(data[ii][2])
            pd.append(data[ii][3])
        width = np.round(1. / len(p), 3)
        acorrs.bar(np.arange(len(p)) + 0 * width, p, width, color='r', label='Pearson R')
        acorrs.bar(np.arange(len(s)) + 1 * width, s, width, color='g', label='Spearman R')
        acorrs.bar(np.arange(len(pd)) + 2 * width, pd, width, color='y', label='Distance R')
        # acorrs.bar(np.arange(len(t))+2*width, t, width, color='b',label='Kendall')
        box = acorrs.get_position()
        acorrs.set_position([box.x0 - 0.065, box.y0 + 0.05, box.width - 0.05, box.height])
        acorrs.legend(framealpha=0.1, loc='center left', bbox_to_anchor=(1, 0.5))
        acorrs.set_xticks(np.arange(len(p)) + 1. * width)
        acorrs.set_xticklabels(labels)
        acorrs.grid(False)
        acorrs.set_ylim([0, 1])
        acorrs.set_xlim([-0.5, len(p)+0.])
        self.canvas = FigureCanvas(f)
        self.hbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.window.add(self.hbox)
        sw = Gtk.ScrolledWindow()
        sw.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        self.hbox.pack_start(sw, True, True, 0)
        sw.add_with_viewport(self.canvas)
        toolbar = NavigationToolbar(self.canvas, self.window)
        self.hbox.pack_start(toolbar, False, True, 0)

        statbar = Gtk.Statusbar()
        self.hbox.pack_start(statbar, False, True, 0)
        self.window.show_all()
        Gtk.main()


class popupWindowWithBoxPlot(object):
    def on_click_me_clicked(self, button=None, widget=None):
        self.window.destroy()
        Gtk.main_quit()
        return True

    def __init__(self, Object=None, data={}):
        self.window = Gtk.Window()
        self.window.set_title("iFAS - Box plot per reference")
        self.window.set_default_size(800, 500)
        self.window.connect("delete-event", self.on_click_me_clicked)
        self.window.connect("destroy", self.on_click_me_clicked)
        p = []
        s = []
        # t = []
        pd = []
        labels = []
        for ii in sorted(data):
            labels.append(ii.split('_')[0])
            p.append(data[ii][:, 0])
            s.append(data[ii][:, 1])
            # t.append(data[ii][:, 2])
            pd.append(data[ii][:, 3])
        self.p = np.transpose(np.asarray(p))
        self.s = np.transpose(np.asarray(s))
        # self.t = np.transpose(np.asarray(t))
        self.pd = np.transpose(np.asarray(pd))
        width = np.round(1. / (len(self.p[0, :]) - 1), 3)
        f = Figure()
        acorrs = f.add_subplot(111)
        acorrs.tick_params(labelsize=16)
        acorrs.bar([0], [0], width, color='r', label='Pearson R')
        acorrs.bar([0], [0], width, color='g', label='Spearman R')
        acorrs.bar([0], [0], width, color='y', label='Distance R')
        # acorrs.bar([0], [0], width, color='b', label='Kendall')
        acorrs.legend(framealpha=0.1, loc='center left', bbox_to_anchor=(1, 0.5))
        acorrs.boxplot(np.abs(self.p), positions=np.arange(len(self.p[0, :])) + 0 * width,\
                       boxprops=dict(color='r', linewidth=3, markersize=12), widths=width)
        acorrs.boxplot(np.abs(self.s), positions=np.arange(len(self.s[0, :])) + 1 * (width + 0.075),\
                       boxprops=dict(color='g', linewidth=3, markersize=12), widths=width)
        acorrs.boxplot(np.abs(self.pd), positions=np.arange(len(self.pd[0, :])) + 2 * (width + 0.075),\
                       boxprops=dict(color='y', linewidth=3, markersize=12), widths=width)
        # acorrs.boxplot(np.abs(self.t), positions=np.arange(len(self.t[0, :])) + 2 * width, boxprops =\
        # 	dict(color='b', linewidth=3, markersize=12), widths=width)
        box = acorrs.get_position()
        acorrs.set_position([box.x0 - 0.065, box.y0 + 0.05, box.width - 0.05, box.height])
        acorrs.set_xticks(np.arange(len(self.p[0, :])) + 1. * width)
        acorrs.set_ylim([0, 1])
        acorrs.set_xlim([-0.5, len(p) + 0.])
        acorrs.set_xticklabels(labels)
        self.canvas = FigureCanvas(f)
        self.hbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.window.add(self.hbox)
        sw = Gtk.ScrolledWindow()
        sw.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        self.hbox.pack_start(sw, True, True, 0)
        sw.add_with_viewport(self.canvas)
        toolbar = NavigationToolbar(self.canvas, self.window)
        self.hbox.pack_start(toolbar, False, True, 0)

        statbar = Gtk.Statusbar()
        self.hbox.pack_start(statbar, False, True, 0)
        plt.rcParams.update({'font.size': 20})
        self.window.show_all()
        pvalue_p, pvalue_adj_p, Rj_p = myUtilities.multiple_comparisons(np.abs(self.p), type='F', ranks=True)
        pvalue_s, pvalue_adj_s, Rj_s = myUtilities.multiple_comparisons(np.abs(self.s), type='F', ranks=True)
        pvalue_pd, pvalue_adj_pd, Rj_pd = myUtilities.multiple_comparisons(np.abs(self.pd), type='F', ranks=True)
        # pvalue_t, pvalue_adj_t, Rj_t = myUtilities.multiple_comparisons(np.abs(self.t), type='F', ranks=True)
        star_message = ''
        max_arg_p = np.argmin(Rj_p)
        max_arg_s = np.argmin(Rj_s)
        max_arg_pd = np.argmin(Rj_pd)
        # max_arg_t = np.argmin(Rj_t)
        for ii in labels:
            star_message += ii + '\n'
        message = create_p_values_string(pvalue_p, 'Pearson R', star_message)
        Object.printMessage(message)
        message = create_p_values_string(pvalue_s, 'Spearman R', star_message)
        Object.printMessage(message)
        message = create_p_values_string(pvalue_pd, 'Distance R', star_message)
        Object.printMessage(message)
        # message = create_p_values_string(pvalue_t, 'Kendalltau', star_message)
        # Object.printMessage(message)
        message = create_bestp_values_string(pvalue_p, 'Pearson R', labels, max_arg_p)
        Object.printMessage(message)
        message = create_bestp_values_string(pvalue_s, 'Spearman R', labels, max_arg_s)
        Object.printMessage(message)
        message = create_bestp_values_string(pvalue_pd, 'Distance R', labels, max_arg_pd)
        Object.printMessage(message)
        # message = create_bestp_values_string(pvalue_t, 'Kendalltau', labels, max_arg_t)
        # Object.printMessage(message)
        Gtk.main()


def create_p_values_string(pvalue, name, star_message):
    message = 'pvalues - ' + name + '\n' + star_message
    for ii in range(pvalue.shape[0]):
        for jj in range(pvalue.shape[1]):
            message += "%.3f\t" % pvalue[ii, jj]
        message += '\n'
    return message


def create_bestp_values_string(pvalue, name, labels, max_arg_p):
    message = 'Best method according to pairwise comparison between' + name + '\n'
    message += labels[max_arg_p] + ' and it is statistically significant better than:\n'
    for jj in range(pvalue.shape[1]):
        if pvalue[max_arg_p, jj] <= 0.05:
            message += labels[jj] + " p_value: %.5f\t" % pvalue[max_arg_p, jj] + '\n'
    return message


class popupWindowWithScatterPlot(object):
    def on_click_me_clicked(self, button=None, widget=None):
        self.window.destroy()
        Gtk.main_quit()
        return True

    def __init__(self, Object=None, data={}, name_axis=()):
        self.window = Gtk.Window()
        self.window.set_title("iFAS - Scatter plot per reference")
        self.window.set_default_size(800, 500)
        self.window.connect("delete-event", self.on_click_me_clicked)
        self.window.connect("destroy", self.on_click_me_clicked)
        f = Figure()
        scplot = f.add_subplot(111)
        scplot.tick_params(labelsize=16)
        gradient = np.linspace(0, 1, len(data))
        cmap = plt.get_cmap('jet')
        colors = cmap(gradient)
        count = 0
        for ii in data:
            scplot.plot(data[ii][:, 0], data[ii][:, 1], color=colors[count, :], label=ii, ls='None', \
                        marker='o', fillstyle='full', ms=10)
            count += 1
        scplot.legend(framealpha=0.1, loc='center left', bbox_to_anchor=(1, 0.5))
        box = scplot.get_position()
        scplot.set_xlabel(name_axis[0], fontsize=16)
        scplot.set_ylabel(name_axis[1], fontsize=16)
        scplot.set_position([box.x0 - 0.065, box.y0 + 0.05, box.width - 0.05, box.height])
        self.canvas = FigureCanvas(f)
        self.hbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.window.add(self.hbox)
        sw = Gtk.ScrolledWindow()
        sw.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        self.hbox.pack_start(sw, True, True, 0)
        sw.add_with_viewport(self.canvas)
        toolbar = NavigationToolbar(self.canvas, self.window)
        self.hbox.pack_start(toolbar, False, True, 0)

        statbar = Gtk.Statusbar()
        self.hbox.pack_start(statbar, False, True, 0)
        self.window.show_all()
        Gtk.main()


class popupWindowWithScatterPlotRegression(object):
    def on_click_me_clicked(self, button=None, widget=None):
        self.window.destroy()
        Gtk.main_quit()
        return True

    def __init__(self, Object=None, data={}, name_axis=()):
        self.window = Gtk.Window()
        self.window.set_title("iFAS - Scatter plot for regression analysis")
        self.window.set_default_size(800, 800)
        self.window.connect("delete-event", self.on_click_me_clicked)
        self.window.connect("destroy", self.on_click_me_clicked)
        f = Figure()
        scplot = f.add_subplot(111)
        scplot.tick_params(labelsize=16)
        scplot.plot(data['t'][:, 0], data['t'][:, 1], color='r', label='Training data', ls='None', \
                    marker='x', fillstyle='full', ms=8, mew=3)
        scplot.plot(data['s'][:, 0], data['s'][:, 1], color='g', label='Testing data', ls='None', \
                    marker='x', fillstyle='full', ms=8, mew=3)
        scplot.plot(data['rl'][:, 0], data['rl'][:, 1], color='b', label='Regression line', ls='-', lw=3)
        scplot.legend(framealpha=0.1, loc='center left', bbox_to_anchor=(1, 0.5))
        box = scplot.get_position()
        scplot.set_xlabel(name_axis[0], fontsize=16)
        scplot.set_ylabel(name_axis[1], fontsize=16)
        scplot.set_position([box.x0 - 0.065, box.y0 + 0.05, box.width - 0.05, box.height])
        self.canvas = FigureCanvas(f)
        self.hbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.window.add(self.hbox)
        sw = Gtk.ScrolledWindow()
        sw.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        self.hbox.pack_start(sw, True, True, 0)
        sw.add_with_viewport(self.canvas)
        toolbar = NavigationToolbar(self.canvas, self.window)
        self.hbox.pack_start(toolbar, False, True, 0)

        statbar = Gtk.Statusbar()
        self.hbox.pack_start(statbar, False, True, 0)
        self.window.show_all()
        Gtk.main()


class popup_window_with_box_plot_differences(object):
    def on_click_me_clicked(self, button=None, widget=None):
        self.window.destroy()
        Gtk.main_quit()
        return True

    def __init__(self, Object=None, data={}, ref_names=[], list_methods=[]):
        self.window = Gtk.Window()
        self.window.set_title("iFAS - Box plot of differences per reference")
        self.window.set_default_size(800, 500)
        self.window.connect("delete-event", self.on_click_me_clicked)
        self.window.connect("destroy", self.on_click_me_clicked)
        labels_ticks = []  # Name of the reference
        X = {}
        for ii in ref_names:
            if not X.has_key(ii):
                X[ii] = {}
            labels_ticks.append(ii)
            for jj in data[ii]:
                for kk in list_methods:
                    if not X[ii].has_key(kk):
                        X[ii][kk] = []
                    X[ii][kk].append(data[ii][jj][kk])
        D = {}
        for ii in list_methods:
            val = []
            for jj in ref_names:
                if not D.has_key(ii):
                    D[ii] = np.array(X[jj][ii])
                else:
                    D[ii] = np.vstack((D[ii], np.array(X[jj][ii])))
            D[ii] = np.transpose(D[ii])
            D[ii] = 1. - D[ii] / np.max(D[ii], 0)
        width = np.round(0.5 / len(D), 3)
        f = Figure()
        diffbox = f.add_subplot(111)
        diffbox.tick_params(labelsize=16)
        gradient = np.linspace(0, 1, len(list_methods))
        cmap = plt.get_cmap('jet')
        colors = cmap(gradient)
        count = 0
        p_values = {}
        Rj = {}
        for ii in list_methods:
            diffbox.bar([0], [0], width, color=colors[count, :], label=ii)
            diffbox.boxplot(D[ii], positions=np.arange(len(D[ii][0, :])) + count * width, boxprops= \
                dict(color=colors[count, :]), widths=width)
            if count == 0:
                diffbox.set_xticks(np.arange(len(D[ii][0, :])) + 1.5 * width)
            p_values[ii], _, Rj[ii] = myUtilities.multiple_comparisons(D[ii], type='F', ranks=True)
            count += 1
        for ii in list_methods:
            start_message = ''
            max_arg = np.argmin(Rj[ii])
            start_message += ii + '\n'
            for jj in ref_names:
                start_message += jj + '\n'
            message = create_p_values_string(p_values[ii], ' ' + ii, start_message)
            Object.print_message(message)
            message = create_bestp_values_string(p_values[ii], ' ' + ii, ref_names, max_arg)
            Object.print_message(message)
        diffbox.legend(framealpha=0.1, loc='center left', bbox_to_anchor=(1, 0.5))
        box = diffbox.get_position()
        diffbox.set_position([box.x0 - 0.065, box.y0 + 0.05, box.width - 0.05, box.height])
        diffbox.set_xticklabels(labels_ticks)
        self.canvas = FigureCanvas(f)
        self.hbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.window.add(self.hbox)
        sw = Gtk.ScrolledWindow()
        sw.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        self.hbox.pack_start(sw, True, True, 0)
        sw.add_with_viewport(self.canvas)
        toolbar = NavigationToolbar(self.canvas, self.window)
        self.hbox.pack_start(toolbar, False, True, 0)

        statbar = Gtk.Statusbar()
        self.hbox.pack_start(statbar, False, True, 0)
        self.window.show_all()
        Gtk.main()


class popupWindowWithContentHistogram(object):
    def on_click_me_clicked(self, button=None, widget=None):
        self.window.destroy()
        Gtk.main_quit()
        return True

    def __init__(self, Object=None, data={}):
        self.window = Gtk.Window()
        self.window.set_title("iFAS - Content histogram per source")
        self.window.set_default_size(800, 500)
        self.window.connect("delete-event", self.on_click_me_clicked)
        self.window.connect("destroy", self.on_click_me_clicked)
        f = Figure()
        histoplot = f.add_subplot(111)
        histoplot.hist(np.asarray(data).ravel(), np.int_(np.sqrt(len(data))), histtype='bar', rwidth=0.5)
        histoplot.tick_params(labelsize=16)
        box = histoplot.get_position()
        histoplot.set_position([box.x0 - 0.065, box.y0 + 0.05, box.width - 0.05, box.height])
        self.canvas = FigureCanvas(f)
        self.hbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.window.add(self.hbox)
        sw = Gtk.ScrolledWindow()
        sw.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        self.hbox.pack_start(sw, True, True, 0)
        sw.add_with_viewport(self.canvas)
        toolbar = NavigationToolbar(self.canvas, self.window)
        self.hbox.pack_start(toolbar, False, True, 0)
        self.window.show_all()
        Gtk.main()


class popupWindowWithHeatParameterMap(object):
    def on_click_me_clicked(self, button=None, widget=None):
        self.window.destroy()
        Gtk.main_quit()
        return True

    def __init__(self, Object=None, data={}, name_axis=('a', 'b'), para_rangex=(0.0, 1), para_rangey=(0.0, 1), step=0.05):
        self.window = Gtk.Window()
        self.window.set_title("iFAS - Performance in function of parameters [Linear combination]")
        self.window.set_default_size(800, 500)
        self.window.connect("delete-event", self.on_click_me_clicked)
        self.window.connect("destroy", self.on_click_me_clicked)
        f = Figure()
        heatmapPCC = f.add_subplot(221)
        heatmapSROCC = f.add_subplot(222)
        heatmapCCD = f.add_subplot(223)
        # heatmaptau = f.add_subplot(224)
        heatmapPCC.tick_params(labelsize=16)
        heatmapSROCC.tick_params(labelsize=16)
        heatmapCCD.tick_params(labelsize=16)
        # heatmaptau.tick_params(labelsize=16)
        rangeparx = np.arange(para_rangex[0], para_rangex[1]+step, step)
        rangepary = np.arange(para_rangey[0], para_rangey[1]+step, step)
        Numx = len(rangeparx)
        Numy = len(rangepary)
        P = np.zeros((Numy, Numx))
        S = np.zeros((Numy, Numx))
        PD = np.zeros((Numy, Numx))
        T = np.zeros((Numy, Numx))
        for aa in range(Numy):
            for bb in range(Numx):
                total = rangeparx[bb] * data[name_axis[0]] + rangepary[aa] * data[name_axis[1]]
                if not (aa == 0 and bb == 0):
                    P[aa, bb], S[aa, bb], T[aa, bb], PD[aa, bb] =\
                        myUtilities.compute_1dcorrelatiosn(total, data[name_axis[2]])
        heatmapPCC.imshow(np.abs(P), extent=[para_rangex[0], para_rangex[1], para_rangey[0], para_rangey[1]],\
                          vmin=0., vmax=1., interpolation='none', aspect='equal', origin='lower', cmap='inferno')
        heatmapSROCC.imshow(np.abs(S), extent=[para_rangex[0], para_rangex[1], para_rangey[0], para_rangey[1]],\
                            vmin=0., vmax=1., interpolation='none', aspect='equal', origin='lower', cmap='inferno')
        axxes = heatmapCCD.imshow(np.abs(PD), extent=[para_rangex[0], para_rangex[1], para_rangey[0], para_rangey[1]],\
                          vmin=0., vmax=1., interpolation='none', aspect='equal', origin='lower', cmap='inferno')
        # axxes = heatmaptau.imshow(np.abs(T), extent=[para_rangex[0], para_rangex[1], para_rangey[0], para_rangey[1]],\
        #                           vmin=0., vmax=1., interpolation='none', aspect='equal', origin='lower', cmap='inferno')
        heatmapPCC.set_xlabel(name_axis[0], fontsize=16)
        heatmapPCC.set_ylabel(name_axis[1], fontsize=16)
        heatmapSROCC.set_xlabel(name_axis[0], fontsize=16)
        heatmapSROCC.set_ylabel(name_axis[1], fontsize=16)
        heatmapCCD.set_xlabel(name_axis[0], fontsize=16)
        heatmapCCD.set_ylabel(name_axis[1], fontsize=16)
        # heatmaptau.set_xlabel(name_axis[0], fontsize=16)
        # heatmaptau.set_ylabel(name_axis[1], fontsize=16)
        cax = f.add_axes([0.925, 0.1, 0.03, 0.8])
        cbar = f.colorbar(axxes, cax=cax)
        cbar.ax.tick_params(labelsize=16)
        heatmapPCC.set_title('PCC', fontsize=16)
        heatmapSROCC.set_title('SROCC', fontsize=16)
        heatmapCCD.set_title('CCD', fontsize=16)
        # heatmaptau.set_title('Tau', fontsize=16)
        #heatmapPCC.set_position([box.x0 - 0.065, box.y0 + 0.05, box.width - 0.05, box.height])
        self.canvas = FigureCanvas(f)
        self.hbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.window.add(self.hbox)
        sw = Gtk.ScrolledWindow()
        sw.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        self.hbox.pack_start(sw, True, True, 0)
        sw.add_with_viewport(self.canvas)
        toolbar = NavigationToolbar(self.canvas, self.window)
        self.hbox.pack_start(toolbar, False, True, 0)

        statbar = Gtk.Statusbar()
        self.hbox.pack_start(statbar, False, True, 0)
        self.window.show_all()
        Gtk.main()


class popupWindowWithScatterPlotWizard(object):
    def on_click_me_clicked(self, button=None, widget=None):
        self.window.destroy()
        Gtk.main_quit()
        return True

    def __init__(self, Object=None, data=np.array([0, 0, 0]), name_axis=('x', 'y')):
        self.window = Gtk.Window()
        self.window.set_title("iFAS - Scatter plot per reference")
        self.window.set_default_size(800, 500)
        self.window.connect("delete-event", self.on_click_me_clicked)
        self.window.connect("destroy", self.on_click_me_clicked)
        f = Figure()
        scplot = f.add_subplot(111)
        scplot.tick_params(labelsize=16)
        distypes = np.unique(data[:, 2])
        gradient = np.linspace(0, 1, distypes.size)
        cmap = plt.get_cmap('jet')
        colors = cmap(gradient)
        count = 0
        for ii in distypes:
            idx = np.where(data[:, 2] == ii)
            scplot.plot(data[idx, 0], data[idx, 1], color=colors[count, :], ls='None', \
                        marker='o', fillstyle='full', ms=10)
            scplot.plot(data[idx[0], 0], data[idx[0], 1], color=colors[count, :], ls='None',\
                        marker='o', fillstyle='full', ms=10, label=str(int(ii)))
            count += 1
        scplot.legend(framealpha=0.1, loc='center left', bbox_to_anchor=(1, 0.5))
        box = scplot.get_position()
        scplot.set_xlabel(name_axis[0], fontsize=16)
        scplot.set_ylabel(name_axis[1], fontsize=16)
        scplot.set_position([box.x0 - 0.065, box.y0 + 0.05, box.width - 0.05, box.height])
        self.canvas = FigureCanvas(f)
        self.hbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.window.add(self.hbox)
        sw = Gtk.ScrolledWindow()
        sw.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        self.hbox.pack_start(sw, True, True, 0)
        sw.add_with_viewport(self.canvas)
        toolbar = NavigationToolbar(self.canvas, self.window)
        self.hbox.pack_start(toolbar, False, True, 0)

        statbar = Gtk.Statusbar()
        self.hbox.pack_start(statbar, False, True, 0)
        self.window.show_all()
        Gtk.main()


def createMenuBar(Object):  # Only for iFAS
    place = Object.builder.get_object("hbox1")
    menubar = Gtk.MenuBar()
    menubar.set_hexpand(True)
    place.pack_start(menubar, True, True, 0)

    # File menu
    filemenu = Gtk.Menu()
    filem = Gtk.MenuItem("File")
    filem.set_submenu(filemenu)
    filem.get_child().modify_font(Pango.FontDescription("Sans 12"))

    menubar.append(filem)

    nmenu = Gtk.Menu()

    nmportm = Gtk.MenuItem("New")
    nmportm.set_submenu(nmenu)
    nmportm.get_child().modify_font(Pango.FontDescription("Sans 12"))

    smenu = Gtk.Menu()

    nsingle = Gtk.MenuItem("Single reference")
    nsingle.set_submenu(smenu)
    nsingle.get_child().modify_font(Pango.FontDescription("Sans 12"))
    ssingle = Gtk.MenuItem("Single processed")
    ssingle.connect("activate", Object.single_ref_single_pro)
    ssingle.get_child().modify_font(Pango.FontDescription("Sans 12"))
    smultiple = Gtk.MenuItem("Multiple processed")
    smultiple.connect("activate", Object.single_ref_multiple_pro)
    smultiple.get_child().modify_font(Pango.FontDescription("Sans 12"))
    smenu.append(ssingle)
    smenu.append(smultiple)

    nmultiple = Gtk.MenuItem("Multiple reference - Multiple processed")
    nmultiple.connect("activate", Object.multiple_ref_multiple_pro)
    nmultiple.get_child().modify_font(Pango.FontDescription("Sans 12"))
    nmenu.append(nsingle)
    nmenu.append(nmultiple)

    filemenu.append(nmportm)

    load = Gtk.MenuItem("Load data")
    load.connect("activate", Object.on_load_data)
    load.get_child().modify_font(Pango.FontDescription("Sans 12"))
    filemenu.append(load)

    save = Gtk.MenuItem("Save data")
    save.connect("activate", Object.on_save_data)
    save.get_child().modify_font(Pango.FontDescription("Sans 12"))
    filemenu.append(save)

    mos = Gtk.MenuItem("Load precomputed data")
    mos.connect("activate", Object.add_precomputed_measure)
    mos.get_child().modify_font(Pango.FontDescription("Sans 12"))
    filemenu.append(mos)

    exit = Gtk.MenuItem("Quit")
    exit.connect("activate", Gtk.main_quit)
    exit.get_child().modify_font(Pango.FontDescription("Sans 12"))
    filemenu.append(exit)

    # Display menu
    dispmenu = Gtk.Menu()
    dispm = Gtk.MenuItem("Display")
    dispm.set_submenu(dispmenu)
    dispm.get_child().modify_font(Pango.FontDescription("Sans 12"))

    idmportm = Gtk.MenuItem("Difference map")
    idmportm.connect("activate", Object.on_change_diff_map)
    idmportm.get_child().modify_font(Pango.FontDescription("Sans 12"))
    timportm = Gtk.MenuItem("Change display images")
    timportm.connect("activate", Object.on_image_set_changed)
    timportm.get_child().modify_font(Pango.FontDescription("Sans 12"))

    dispmenu.append(idmportm)
    dispmenu.append(timportm)

    menubar.append(dispm)

    # Plots menu
    plotmenu = Gtk.Menu()
    plots = Gtk.MenuItem("Plots")
    plots.set_submenu(plotmenu)
    plots.get_child().modify_font(Pango.FontDescription("Sans 12"))
    MxMy = Gtk.MenuItem("Mx Vs My")
    MxMy.get_child().modify_font(Pango.FontDescription("Sans 12"))
    MxMy.connect("activate", Object.plot_x_y)
    MxMyall = Gtk.MenuItem("Mx Vs My [full data]")
    MxMyall.get_child().modify_font(Pango.FontDescription("Sans 12"))
    MxMyall.connect("activate", Object.on_scatter_plot)
    wizard = Gtk.MenuItem("Wizard: Multiple distortion")
    wizard.get_child().modify_font(Pango.FontDescription("Sans 12"))
    wizard.connect("activate", Object.on_multiple_distortion_plot)
    plotmenu.append(MxMy)
    plotmenu.append(MxMyall)
    plotmenu.append(wizard)
    menubar.append(plots)

    # correlation analysis
    corranal = Gtk.Menu()
    cmenu = Gtk.MenuItem("Correlation analysis")
    cmenu.get_child().modify_font(Pango.FontDescription("Sans 12"))
    cmenu.set_submenu(corranal)
    barplot = Gtk.MenuItem("Global bar plot")
    barplot.get_child().modify_font(Pango.FontDescription("Sans 12"))
    barplot.connect("activate", Object.global_correlation_bar_plot)
    boxplot = Gtk.MenuItem("Per source box plot")
    boxplot.get_child().modify_font(Pango.FontDescription("Sans 12"))
    boxplot.connect("activate", Object.box_plot_reference)
    heatmap = Gtk.MenuItem("Heat map to combine features")
    heatmap.get_child().modify_font(Pango.FontDescription("Sans 12"))
    heatmap.connect("activate", Object.on_heat_map)
    corranal.append(barplot)
    corranal.append(boxplot)
    corranal.append(heatmap)
    menubar.append(cmenu)

    # regression analysis
    reganal = Gtk.Menu()
    rmenu = Gtk.MenuItem("Regression analysis")
    rmenu.get_child().modify_font(Pango.FontDescription("Sans 12"))
    rmenu.set_submenu(reganal)
    linear = Gtk.MenuItem("Linear")
    linear.get_child().modify_font(Pango.FontDescription("Sans 12"))
    linear.connect("activate", Object.on_regression, 'linear')
    quad = Gtk.MenuItem("Quadratic")
    quad.get_child().modify_font(Pango.FontDescription("Sans 12"))
    quad.connect("activate", Object.on_regression, 'quadratic')
    cubic = Gtk.MenuItem("Cubic")
    cubic.get_child().modify_font(Pango.FontDescription("Sans 12"))
    cubic.connect("activate", Object.on_regression, 'cubic')
    expo = Gtk.MenuItem("Exponential")
    expo.get_child().modify_font(Pango.FontDescription("Sans 12"))
    expo.connect("activate", Object.on_regression, 'exponential')
    logis = Gtk.MenuItem("Logistic")
    logis.get_child().modify_font(Pango.FontDescription("Sans 12"))
    logis.connect("activate", Object.on_regression, 'logistic')
    cerro = Gtk.MenuItem("Complementary error")
    cerro.get_child().modify_font(Pango.FontDescription("Sans 12"))
    cerro.connect("activate", Object.on_regression, 'complementaryError')
    reganal.append(linear)
    reganal.append(quad)
    reganal.append(cubic)
    reganal.append(expo)
    reganal.append(logis)
    reganal.append(cerro)
    menubar.append(rmenu)


    # Tools menu
    toolmenu = Gtk.Menu()
    toolsm = Gtk.MenuItem("Tools")
    toolsm.set_submenu(toolmenu)
    toolsm.get_child().modify_font(Pango.FontDescription("Sans 12"))

    conthist = Gtk.MenuItem("Content features per source")
    conthist.connect("activate", Object.histogram_content_features)
    conthist.get_child().modify_font(Pango.FontDescription("Sans 12"))

    toolmenu.append(conthist)

    menubar.append(toolsm)

    # Package management
    managemenu = Gtk.Menu()
    manamenu = Gtk.MenuItem("Manage packages")
    manamenu.set_submenu(managemenu)
    manamenu.get_child().modify_font(Pango.FontDescription("Sans 12"))

    addfg = Gtk.MenuItem("Add package")
    addfg.connect("activate", Object.add_new_python_script)
    addfg.get_child().modify_font(Pango.FontDescription("Sans 12"))

    mmportm = Gtk.MenuItem("Available measures")
    mmportm.connect("activate", Object.on_select_measures)
    mmportm.get_child().modify_font(Pango.FontDescription("Sans 12"))

    gmportm = Gtk.MenuItem("Available packages")
    gmportm.connect("activate", Object.on_select_package)
    gmportm.get_child().modify_font(Pango.FontDescription("Sans 12"))

    managemenu.append(gmportm)
    managemenu.append(mmportm)
    managemenu.append(addfg)
    menubar.append(manamenu)

    # Help menu
    helpmenu = Gtk.Menu()
    helpm = Gtk.MenuItem("Help")
    helpm.set_submenu(helpmenu)
    helpm.get_child().modify_font(Pango.FontDescription("Sans 12"))

    guide = Gtk.MenuItem("Guide")
    guide.connect("activate", Object.on_guide_clicked)
    guide.get_child().modify_font(Pango.FontDescription("Sans 12"))
    about = Gtk.MenuItem("About")
    about.connect("activate", Object.on_about_click)
    about.get_child().modify_font(Pango.FontDescription("Sans 12"))
    screenupdate = Gtk.MenuItem("Refresh screen")
    screenupdate.connect("activate", Object.update_screen_size)
    screenupdate.get_child().modify_font(Pango.FontDescription("Sans 12"))
    helpmenu.append(guide)
    helpmenu.append(about)
    helpmenu.append(screenupdate)
    menubar.append(helpm)
