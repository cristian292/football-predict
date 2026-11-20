"""Reusable GTK widgets; tables keep aligned, selectable numeric columns."""
from gi.repository import Gtk


def label(text='', style=None):
    widget = Gtk.Label(label=text, xalign=0, wrap=True, selectable=False)
    if style:
        widget.add_css_class(style)
    return widget


def section(title, child):
    box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
    box.add_css_class('panel')
    box.append(label(title, 'heading'))
    box.append(child)
    return box


class DataTable(Gtk.Box):
    def __init__(self, columns):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.columns = columns
        self.rows = []
        self.add_css_class('data-table')
        self.set_rows([])

    def set_rows(self, rows, columns=None):
        if columns is not None:
            self.columns = columns
        self.rows = [[str(value) for value in row] for row in rows]
        child = self.get_first_child()
        while child:
            following = child.get_next_sibling()
            self.remove(child)
            child = following
        grid = Gtk.Grid(column_spacing=22, row_spacing=0, hexpand=True)
        for row_index, row in enumerate([self.columns] + self.rows):
            for column_index, value in enumerate(row):
                cell = Gtk.Label(label=str(value), xalign=0 if column_index == 0 else 1, selectable=True, hexpand=True)
                cell.add_css_class('table-head' if row_index == 0 else 'table-cell')
                if column_index:
                    cell.add_css_class('numeric')
                grid.attach(cell, column_index, row_index, 1, 1)
        self.append(grid)
        if not self.rows:
            self.append(label('Nu există încă rezultate.', 'muted'))
