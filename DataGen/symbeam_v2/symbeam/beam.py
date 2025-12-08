"""Beam module.

SymBeam main module. It contains the beam class definition, which is the main object
of any analysis, allowing for the symbolic solution of equilibirum equations, internal
shear force and bending moment diagram computation and deflection analysis.

Furthermore, the beam class is endowed with all plotting and output capabilities that
SymBeam furnishes to the outside world.

..module:: beam
  :synopsis: Main beam class

..moduleauthor:: A. M. Couto Carneiro <amcc@fe.up.pt>
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sympy as sym

from sympy.abc import E, I, x

from symbeam.load import distributed_load, point_load, point_moment
from symbeam.point import continuity, fixed, hinge, pin, roller

# ---- new imports ----
import json
# ---- /new imports ----


# Set numerical tolerance
tol = 1e-6
# ===================================================================================== beam
class beam:
    """Beam main class.
    This class provides and interface for all functionalily available in the package: beam
    solving and plotting.
    """

    def __init__(self, length, x0=0):
        """Beam class constructor.

        Parameters
        ----------
        length : sympifiable type (int, float, string, SympP symbol, etc)
          Length of the beam
        x0 : sympifiable type (int, float, string, SympP symbol, etc)
          Initial point of the beam

        Notes
        -----
        The beam can be instantiated either with numeric our symbolic input. For instance,
        the length o the beam can be 1 or 'L'. Whenver a symbols is used for the length,
        SymBeam only accepts coordinates expressed as a definite fraction of 'L', such as
        'L/4'. If more symbols are introduce in the geometry definition, e.g., 'a*L', the
        program cannot complete, as several solutions could arise to the problem.
        """
        # Initialise the length and startig point of the beam.
        self.length = sym.sympify(length)
        self.x0 = sym.sympify(x0)

        # Check the consitency of the input. Only one symbol is permitted for the geometry
        # definition, in order to facilitate the creation of the segments.
        # If the beam starts at zero it's fine.
        starts_at_zero = self.x0 == sym.sympify(0)
        if len(self.length.free_symbols) != len(self.x0.free_symbols) and not (
            starts_at_zero
        ):
            raise RuntimeError(
                "The number of symbols set for the length and initial "
                + "beam coordinate is distinct. Only one symbol is allowed for the "
                + "definition of the geometry"
            )

        # Make sure the initial position and length use the same symbol.
        if len(self.length.free_symbols) == 1 and not (starts_at_zero):
            if next(iter(self.length.free_symbols)) != next(iter(self.x0.free_symbols)):
                raise RuntimeError(
                    "The length and initial coordinate of the beam have "
                    + "been defined with distinct symbols."
                )

        if len(self.length.free_symbols) > 1:
            raise RuntimeError(
                "Only one symbols is allowed to define the length of" + " the beam."
            )

        # Store the length symbol
        if len(self.length.free_symbols) == 1:
            self.length_symbol = next(iter(self.length.free_symbols))

        # Initialise the list storing all the input information for the beam
        self.support_list = []
        self.distributed_load_list = []
        self.point_load_list = []
        self.point_moment_list = []
        self.young_segment_list = [_property_segment(self.x0, self.x0 + self.length, E)]
        self.young_default = True
        self.inertia_segment_list = [_property_segment(self.x0, self.x0 + self.length, I)]
        self.inertia_default = True

        # Initialise the processed beam information.
        self.segments = []
        self.points = []

    # -------------------------------------------------------------------------- add_support
    def add_support(self, x_coord, support_type):
        """Appends a new support to the list of input supports.

        Parameters
        ----------
        x_coord : sympifiable type (int, float, string, SympP symbol, etc)
          Coordiante of the support
        support_type : str
          Type of support
        """
        # First check if the coordinate inside the beam.
        self._check_inside_beam(x_coord)

        # Now allocate the correct point type (polymorphism).
        if support_type.lower() == "pin":
            new_point = pin(x_coord)

        elif support_type.lower() == "roller":
            new_point = roller(x_coord)

        elif support_type.lower() == "hinge":
            new_point = hinge(x_coord)

        elif support_type.lower() == "fixed":
            new_point = fixed(x_coord)

        else:
            raise RuntimeError("Unknown support type: {0}.".format(type))

        # If no point exists at that location, create a new one in the beam.
        for point in self.support_list:
            if point.x_coord == new_point.x_coord:
                raise RuntimeError(
                    "Repeated support for x = {0}.".format(new_point.x_coord)
                )

        self.support_list.append(new_point)

    # ----------------------------------------------------------------- add_distributed_load
    def add_distributed_load(self, x_start, x_end, expression):
        """Appends a new distributed load to the list of input distributed loads.

        Parameters
        ----------
        x_start : sympifiable type (int, float, string, SympP symbol, etc)
          Starting coordinate of the distributed load
        x_end : sympifiable type (int, float, string, SympP symbol, etc)
          Starting coordinate of the distributed load
        expression  : sympifiable type (int, float, string, SympP symbol, etc)
          Distributed loading expression
        """
        self._check_coordinates(x_start, x_end)
        new_load = distributed_load(x_start, x_end, expression)
        self.distributed_load_list.append(new_load)

    # ----------------------------------------------------------------------- add_point_load
    def add_point_load(self, x_coord, value):
        """Appends a new point load to the list of input point loads.

        Parameters
        ----------
        x_coord : sympifiable type (int, float, string, SympP symbol, etc)
          Coordinate of the applied load
        value : sympifiable type (int, float, string, SympP symbol, etc)
          Load value
        """
        self._check_inside_beam(x_coord)
        new_load = point_load(x_coord, value)
        self.point_load_list.append(new_load)

    # --------------------------------------------------------------------- add_point_moment
    def add_point_moment(self, x_coord, value):
        """Appends a new point moment to the list of input point moments.

        Parameters
        ----------
        x_coord : sympifiable type (int, float, string, SympP symbol, etc)
          Coordinate of the applied moment
        value : sympifiable type (int, float, string, SympP symbol, etc)
          Moment value
        """
        self._check_inside_beam(x_coord)
        new_moment = point_moment(x_coord, value)
        self.point_moment_list.append(new_moment)

    # ---------------------------------------------------------------------------- set_young
    def set_young(self, x_start, x_end, value):
        """Sets the young modulus of a portion of the beam.

        Parameters
        ----------
        x_start : sympifiable type (int, float, string, SympP symbol, etc)
          Starting coordinate of the distributed load
        x_end : sympifiable type (int, float, string, SympP symbol, etc)
          Starting coordinate of the distributed load
        expression  : sympifiable type (int, float, string, SympP symbol, etc)
          Young modulus value
        """
        # First check if the coordinate inside the beam.
        self._check_coordinates(x_start, x_end)

        # If the user specifies the property explicitely, reset the default values
        if self.young_default:
            self.young_default = False
            self.young_segment_list = []
        new_young = _property_segment(x_start, x_end, value)
        self.young_segment_list.append(new_young)

    # -------------------------------------------------------------------------- set_inertia
    def set_inertia(self, x_start, x_end, value):
        """Sets the second moment of area of a portion of the beam."""
        # First check if the coordinate inside the beam.
        self._check_coordinates(x_start, x_end)

        # If the user specifies the property explicitely, reset the default values
        if self.inertia_default:
            self.inertia_default = False
            self.inertia_segment_list = []
        new_inertia = _property_segment(x_start, x_end, value)
        self.inertia_segment_list.append(new_inertia)

    # ---------------------------------------------------------------- check_beam_properties
    def _check_beam_properties(self):
        """Verifies is the beam properties have been consistently set along the entire
        length of the beam.
        """
        # If the length is given as a variable, substitute by 1 for ease of comparison.
        if self.length.is_number:
            length_numeric = self.length
            x0_numeric = self.x0
            young_x_start_numeric = [item.x_start for item in self.young_segment_list]
            young_x_end_numeric = [item.x_end for item in self.young_segment_list]
            inertia_x_start_numeric = [item.x_start for item in self.inertia_segment_list]
            inertia_x_end_numeric = [item.x_end for item in self.inertia_segment_list]
        else:
            length_numeric = self.length.subs({self.length_symbol: 1.0})
            x0_numeric = self.x0.subs({self.length_symbol: 1.0})
            young_x_start_numeric = [
                item.x_start.subs({self.length_symbol: 1.0})
                for item in self.young_segment_list
            ]
            young_x_end_numeric = [
                item.x_end.subs({self.length_symbol: 1.0})
                for item in self.young_segment_list
            ]
            inertia_x_start_numeric = [
                item.x_start.subs({self.length_symbol: 1.0})
                for item in self.inertia_segment_list
            ]
            inertia_x_end_numeric = [
                item.x_end.subs({self.length_symbol: 1.0})
                for item in self.inertia_segment_list
            ]

        length_young = 0.0
        length_inertia = 0.0

        # Quick check Young modulus specification (check if the proprty segments sum up
        # to the length of the beam).
        for i in range(len(self.young_segment_list)):
            if young_x_start_numeric[i] < x0_numeric:
                raise RuntimeError(
                    "Yound modulus specified for segment starting outside " + "the beam."
                )

            if young_x_end_numeric[i] > length_numeric + x0_numeric:
                raise RuntimeError(
                    "Young modulus specified for segment ending outside " + "the beam."
                )

            length_young = length_young + young_x_end_numeric[i] - young_x_start_numeric[i]

        if abs(length_young - length_numeric) > tol:
            raise RuntimeError(
                "Inconsistent specification of the Young modulus along "
                + "the beam. There are either repeated or missing segments of the beam."
            )

        # Now a more thorough verification (truly check if the segments are overlapping).
        for i in range(len(self.young_segment_list)):
            for j in range(len(self.young_segment_list)):
                if j != i:
                    is_on_the_right = (
                        young_x_end_numeric[j] < young_x_start_numeric[i] + tol
                    )
                    is_on_the_left = young_x_start_numeric[j] > young_x_end_numeric[i] - tol
                    if not (is_on_the_left or is_on_the_right):
                        raise RuntimeError(
                            "Inconsistent specification of the Young "
                            + "modulus along the beam. There are either repeated or missing "
                            + "segments of the beam."
                        )

        # Quick check second moment of area specification (check if the proprty segments sum
        # up to the length of the beam).
        for i in range(len(self.inertia_segment_list)):
            if inertia_x_start_numeric[i] < x0_numeric:
                raise RuntimeError(
                    "Moment of inertia specified for segment starting "
                    + "outside the beam."
                )

            if inertia_x_end_numeric[i] > length_numeric + x0_numeric:
                raise RuntimeError(
                    "Moment of inertia specified for segment ending " + "outside the beam."
                )

            length_inertia = (
                length_inertia + inertia_x_end_numeric[i] - inertia_x_start_numeric[i]
            )

        if abs(length_inertia - length_numeric) > tol:
            raise RuntimeError(
                "Inconsistent specification of the moment of inertia "
                + "along the beam. There are either repeated or missing segments of the beam."
            )

        # Now a more thorough verification (truly check if the segments are overlapping).
        for i in range(len(self.inertia_segment_list)):
            for j in range(len(self.inertia_segment_list)):
                if j != i:
                    is_on_the_right = (
                        inertia_x_end_numeric[j] < inertia_x_start_numeric[i] + tol
                    )
                    is_on_the_left = (
                        inertia_x_start_numeric[j] > inertia_x_end_numeric[i] - tol
                    )
                    if not (is_on_the_left or is_on_the_right):
                        raise RuntimeError(
                            "Inconsistent specification of the moment of "
                            + "inertia along the beam. There are either repeated or missing "
                            + "segments of the beam."
                        )

    # -------------------------------------------------------------------- check_coordinates
    def _check_coordinates(self, x_start, x_end):
        """Checks if the input segment coordinates are valid.

        Parameters
        ----------
        x_start : sympifiable type (int, float, string, SympP symbol, etc)
          Starting coordinate of the distributed load
        x_end : sympifiable type (int, float, string, SympP symbol, etc)
          Starting coordinate of the distributed load
        """
        x_start_symbol = sym.sympify(x_start)
        x_end_symbol = sym.sympify(x_end)

        if len(x_start_symbol.free_symbols) > 1:
            raise RuntimeError("More than one symbol used for starting coordinate.")

        if len(x_end_symbol.free_symbols) > 1:
            raise RuntimeError("More than one symbol used for coordinate coordinate.")

        if len(x_start_symbol.free_symbols) == 1:
            if next(iter(x_start_symbol.free_symbols)) != self.length_symbol:
                raise RuntimeError(
                    "Distinct symbols have been used for coordinate along "
                    + "the the beam and beam length."
                )

        if len(x_end_symbol.free_symbols) == 1:
            if next(iter(x_end_symbol.free_symbols)) != self.length_symbol:
                raise RuntimeError(
                    "Distinct symbols have been used for coordinate along"
                    + " the the beam and beam length."
                )

        if self.length.is_number:
            x_start_numeric = x_start_symbol
            x_end_numeric = x_end_symbol
        else:
            x_start_numeric = x_start_symbol.subs({self.length_symbol: 1})
            x_end_numeric = x_end_symbol.subs({self.length_symbol: 1})

        if abs(x_start_numeric - x_end_numeric) < tol:
            raise RuntimeError(
                "The specified beam segment is too short. The poits are " + "overlapping."
            )

        if x_start_numeric > x_end_numeric:
            raise RuntimeError(
                "The starting coordinate is greater than the ending " + "coordinte."
            )

    # -------------------------------------------------------------------- check_inside_beam
    def _check_inside_beam(self, x_coord):
        """Chekcs if a given coordinate is valid and lies inside the beam domain.

        Parameters
        ----------
        x_coord : sympifiable type (int, float, string, SympP symbol, etc)
          Coordinate of the point
        """
        x_coord_symbol = sym.sympify(x_coord)

        if len(x_coord_symbol.free_symbols) > 1:
            raise RuntimeError("More than one symbol used for coordinate.")

        if len(x_coord_symbol.free_symbols) == 1:
            if next(iter(x_coord_symbol.free_symbols)) != self.length_symbol:
                raise RuntimeError(
                    "Distinct symbols have been used for coordinate along "
                    + "the the beam and beam length."
                )

        if self.length.is_number:
            x_coord_numeric = x_coord_symbol
            length_numeric = self.length
            x0_numeric = self.x0
        else:
            x_coord_numeric = x_coord_symbol.subs({self.length_symbol: 1})
            length_numeric = self.length.subs({self.length_symbol: 1.0})
            x0_numeric = self.x0.subs({self.length_symbol: 1.0})

        if not (x0_numeric - tol <= x_coord_numeric <= x0_numeric + length_numeric + tol):
            raise RuntimeError("The specified coordinate lies outside the beam.")

    # ------------------------------------------------------------------------- set_segments
    def _set_segments(self):
        """Create the beam segments, such that the properties and loads are piecewise
        continuous within each segments and, therefore, are properly setup for symbolic
        integration.
        """
        # Start by defining all numeric variables in order to be able sort the segments
        if self.length.is_number:
            length_numeric = self.length
            x0_numeric = self.x0
            young_x_start_numeric = [item.x_start for item in self.young_segment_list]
            young_x_end_numeric = [item.x_end for item in self.young_segment_list]
            inertia_x_start_numeric = [item.x_start for item in self.inertia_segment_list]
            inertia_x_end_numeric = [item.x_end for item in self.inertia_segment_list]
            support_x_numeric = [item.x_coord for item in self.support_list]
            point_load_x_numeric = [item.x_coord for item in self.point_load_list]
            point_moment_x_numeric = [item.x_coord for item in self.point_moment_list]
            distributed_x_start_numeric = [
                item.x_start for item in self.distributed_load_list
            ]
            distributed_x_end_numeric = [item.x_end for item in self.distributed_load_list]
        else:
            length_numeric = self.length.subs({self.length_symbol: 1.0})
            x0_numeric = self.x0.subs({self.length_symbol: 1.0})
            young_x_start_numeric = [
                item.x_start.subs({self.length_symbol: 1.0})
                for item in self.young_segment_list
            ]
            young_x_end_numeric = [
                item.x_end.subs({self.length_symbol: 1.0})
                for item in self.young_segment_list
            ]
            inertia_x_start_numeric = [
                item.x_start.subs({self.length_symbol: 1.0})
                for item in self.inertia_segment_list
            ]
            inertia_x_end_numeric = [
                item.x_end.subs({self.length_symbol: 1.0})
                for item in self.inertia_segment_list
            ]
            support_x_numeric = [
                item.x_coord.subs({self.length_symbol: 1.0}) for item in self.support_list
            ]
            point_load_x_numeric = [
                item.x_coord.subs({self.length_symbol: 1.0})
                for item in self.point_load_list
            ]
            point_moment_x_numeric = [
                item.x_coord.subs({self.length_symbol: 1.0})
                for item in self.point_moment_list
            ]
            distributed_x_start_numeric = [
                item.x_start.subs({self.length_symbol: 1.0})
                for item in self.distributed_load_list
            ]
            distributed_x_end_numeric = [
                item.x_end.subs({self.length_symbol: 1.0})
                for item in self.distributed_load_list
            ]

        young_x_start_symbol = [item.x_start for item in self.young_segment_list]
        young_x_end_symbol = [item.x_end for item in self.young_segment_list]
        inertia_x_start_symbol = [item.x_start for item in self.inertia_segment_list]
        inertia_x_end_symbol = [item.x_end for item in self.inertia_segment_list]
        support_x_symbol = [item.x_coord for item in self.support_list]
        point_load_x_symbol = [item.x_coord for item in self.point_load_list]
        point_moment_x_symbol = [item.x_coord for item in self.point_moment_list]
        distributed_x_start_symbol = [item.x_start for item in self.distributed_load_list]
        distributed_x_end_symbol = [item.x_end for item in self.distributed_load_list]

        # Store all coordinates in a single list, sort by ascending order and remove
        # possible duplicate entries.
        all_x_coord_numeric = []
        all_x_coord_numeric.extend(young_x_start_numeric)
        all_x_coord_numeric.extend(young_x_end_numeric)
        all_x_coord_numeric.extend(inertia_x_start_numeric)
        all_x_coord_numeric.extend(inertia_x_end_numeric)
        all_x_coord_numeric.extend(support_x_numeric)
        all_x_coord_numeric.extend(point_load_x_numeric)
        all_x_coord_numeric.extend(point_moment_x_numeric)
        all_x_coord_numeric.extend(distributed_x_start_numeric)
        all_x_coord_numeric.extend(distributed_x_end_numeric)

        all_x_coord_symbol = []
        all_x_coord_symbol.extend(young_x_start_symbol)
        all_x_coord_symbol.extend(young_x_end_symbol)
        all_x_coord_symbol.extend(inertia_x_start_symbol)
        all_x_coord_symbol.extend(inertia_x_end_symbol)
        all_x_coord_symbol.extend(support_x_symbol)
        all_x_coord_symbol.extend(point_load_x_symbol)
        all_x_coord_symbol.extend(point_moment_x_symbol)
        all_x_coord_symbol.extend(distributed_x_start_symbol)
        all_x_coord_symbol.extend(distributed_x_end_symbol)

        # Convert SymPy numeric values to Python floats for sorting
        # This prevents "cannot determine truth value of Relational" errors
        import sympy
        all_x_coord_numeric_float = []
        for val in all_x_coord_numeric:
            if isinstance(val, sympy.Basic):
                # Convert SymPy numeric to Python float
                try:
                    all_x_coord_numeric_float.append(float(val.evalf()))
                except (TypeError, ValueError):
                    # If conversion fails, try direct float conversion
                    all_x_coord_numeric_float.append(float(val))
            else:
                all_x_coord_numeric_float.append(float(val))
        
        # Sort coordinates using only the numeric (float) keys to avoid SymPy comparisons
        sorted_x_coords = sorted(
            zip(all_x_coord_numeric_float, all_x_coord_symbol),
            key=lambda pair: pair[0]
        )
        all_x_coord_symbol = [item for _, item in sorted_x_coords]
        all_x_coord_numeric_float.sort()
        all_x_coord_numeric = all_x_coord_numeric_float

        keep_x_coord = [True]
        for i in range(1, len(all_x_coord_numeric)):
            if abs(all_x_coord_numeric[i] - all_x_coord_numeric[i - 1]) < tol:
                keep_x_coord.extend([False])
            else:
                keep_x_coord.extend([True])

        beam_x_coord = [
            all_x_coord_symbol[i]
            for i in range(len(all_x_coord_numeric))
            if keep_x_coord[i]
        ]
        beam_x_coord_numeric = [
            all_x_coord_numeric[i]
            for i in range(len(all_x_coord_numeric))
            if keep_x_coord[i]
        ]
        if abs(beam_x_coord_numeric[0] - x0_numeric) > tol:
            raise RuntimeError(
                "Error in segment creation: the first x-coordinate does "
                + "not match the initial beam coordinate."
            )

        if abs(beam_x_coord_numeric[-1] - x0_numeric - length_numeric) > tol:
            raise RuntimeError(
                "Error in segment creation: the last x-coordinate is not "
                + "consistent with the length of the beam."
            )

        # Create the list of points of the beam
        # -------------------------------------
        for i in range(len(beam_x_coord)):
            # First, check if the point is a support.
            is_support = False
            support = None
            for j, sup in enumerate(self.support_list):
                if abs(support_x_numeric[j] - beam_x_coord_numeric[i]) < tol:
                    is_support = True
                    support = sup
                    break

            # If the point is a support, create a new instance, otherwise, set it as a
            # continuity point.
            if is_support:
                this_point = type(support)(beam_x_coord[i])
            else:
                this_point = continuity(beam_x_coord[i])

            if self.length.is_number:
                this_x_numeric = this_point.x_coord
            else:
                this_x_numeric = this_point.x_coord.subs({self.length_symbol: 1.0})

            # Second, go add all external point loads and moments.
            for j, load in enumerate(self.point_load_list):
                if abs(point_load_x_numeric[j] - this_x_numeric) < tol:
                    this_point.external_force = this_point.external_force + load.value

            for j, moment in enumerate(self.point_moment_list):
                if abs(point_moment_x_numeric[j] - this_x_numeric) < tol:
                    this_point.external_moment = this_point.external_moment + moment.value

            self.points.append(this_point)

        # Create the list of segments of the beam
        # ---------------------------------------
        for i in range(len(self.points) - 1):
            x_start = self.points[i].x_coord
            x_end = self.points[i + 1].x_coord

            if self.length.is_number:
                x_start_numeric = x_start
                x_end_numeric = x_end
            else:
                x_start_numeric = x_start.subs({self.length_symbol: 1.0})
                x_end_numeric = x_end.subs({self.length_symbol: 1.0})

            # First, find the correct Young modulus segment.
            found_segment = False
            for j in range(len(self.young_segment_list)):
                lower_bound = x_start_numeric > young_x_start_numeric[j] - tol
                upper_bound = x_end_numeric < young_x_end_numeric[j] + tol

                if lower_bound and upper_bound:
                    found_segment = True
                    break

            if not (found_segment):
                raise RuntimeError("Search for valid Young modulus segment failed.")

            young = self.young_segment_list[j].value

            # Second, find the correct inertia segment.
            found_segment = False
            for j in range(len(self.inertia_segment_list)):
                lower_bound = x_start_numeric > inertia_x_start_numeric[j] - tol
                upper_bound = x_end_numeric < inertia_x_end_numeric[j] + tol

                if lower_bound and upper_bound:
                    found_segment = True
                    break

            if not (found_segment):
                raise RuntimeError("Search for valid moment of ienrtia segment failed.")

            inertia = self.inertia_segment_list[j].value

            # Lastly, find the expression of the resultant distributed load.
            q_load_expression = sym.sympify(0)
            for j in range(len(self.distributed_load_list)):
                lower_bound = x_start_numeric > distributed_x_start_numeric[j] - tol
                upper_bound = x_end_numeric < distributed_x_end_numeric[j] + tol

                if lower_bound and upper_bound:
                    q_load_expression = (
                        q_load_expression + self.distributed_load_list[j].expression
                    )

            q_load = distributed_load(x_start, x_end, q_load_expression)

            self.segments.append(_segment(x_start, x_end, q_load, young, inertia))

    # ---------------------------------------------------------------------- solve_reactions
    def _solve_reactions(self):
        """Solves the static equilibirum equations of the beam and determines all the
        support reactions.
        """
        # Check if the beam is solvable.
        num_equilibirum_equations = 3
        num_fixed_degrees_of_freedom = 0
        for i, ipoint in enumerate(self.points):
            num_fixed_degrees_of_freedom = (
                num_fixed_degrees_of_freedom + ipoint.get_fixed_degree_of_freedom()
            )
            if isinstance(ipoint, hinge):
                num_equilibirum_equations = num_equilibirum_equations + 1

        if num_equilibirum_equations > num_fixed_degrees_of_freedom:
            raise RuntimeError(
                "The number of equilibirum equations is greater than "
                + "the number of fixed degrees of freedom: the beam is not statically fixed."
            )

        if num_equilibirum_equations < num_fixed_degrees_of_freedom:
            raise RuntimeError(
                "The number of equilibirum equations is less than "
                + "the number of fixed degrees of freedom: the beam is "
                + "statically indeterminate."
            )

        # Create the symbolic variables denoting the reactions at each support.
        unknown_reactions = []
        reaction_force_points = []
        reaction_moment_points = []
        # Reactions forces first.
        for i in range(len(self.points)):
            if self.points[i].has_reaction_force():
                name = "R{0}".format(i)
                self.points[i].reaction_force = sym.symbols(name)
                unknown_reactions.append(self.points[i].reaction_force)
                reaction_force_points.append(i)

        # Then reaction moments.
        for i in range(len(self.points)):
            if self.points[i].has_reaction_moment():
                name = "M{0}".format(i)
                self.points[i].reaction_moment = sym.symbols(name)
                unknown_reactions.append(self.points[i].reaction_moment)
                reaction_moment_points.append(i)

        # Equilibirum of forces in the y-direction.
        sum_forces_y = sym.sympify(0)

        for ipoint in self.points:
            sum_forces_y = sum_forces_y + ipoint.external_force + ipoint.reaction_force

        for isegment in self.segments:
            sum_forces_y = sum_forces_y + isegment.distributed_load.equivalent_force

        # Equilibirum of moments in the z-direction on the initial point.
        sum_moments_z = sym.sympify(0)

        for ipoint in self.points:
            sum_moments_z = sum_moments_z + ipoint.reaction_moment
            sum_moments_z = sum_moments_z + ipoint.reaction_force * (
                ipoint.x_coord - self.x0
            )
            sum_moments_z = sum_moments_z + ipoint.external_moment
            sum_moments_z = sum_moments_z + ipoint.external_force * (
                ipoint.x_coord - self.x0
            )

        for isegment in self.segments:
            sum_moments_z = sum_moments_z + isegment.distributed_load.equivalent_moment

        # System of global equations.
        equilibirum_equations = [sum_forces_y, sum_moments_z]

        # Added more moment equilibirum equation at the hinges.
        for i, ipoint in enumerate(self.points):
            sum_moments_z_hinge = sym.sympify(0)
            if isinstance(ipoint, hinge):
                for jpoint in self.points[i + 1 :]:
                    sum_moments_z_hinge = sum_moments_z_hinge + jpoint.reaction_moment
                    sum_moments_z_hinge = sum_moments_z_hinge + jpoint.reaction_force * (
                        jpoint.x_coord - ipoint.x_coord
                    )
                    sum_moments_z_hinge = sum_moments_z_hinge + jpoint.external_moment
                    sum_moments_z_hinge = sum_moments_z_hinge + jpoint.external_force * (
                        jpoint.x_coord - ipoint.x_coord
                    )

                for jsegment in self.segments[i:]:
                    sum_moments_z_hinge = (
                        sum_moments_z_hinge
                        + jsegment.distributed_load.equivalent_moment
                        - jsegment.distributed_load.equivalent_force * ipoint.x_coord
                    )

                equilibirum_equations.extend([sum_moments_z_hinge])

        # Solve the system of equations.
        if len(equilibirum_equations) != len(unknown_reactions):
            raise RuntimeError(
                "[Internal error: the number of equilibirum equations "
                + "and unknwown reactions does nto match. Check the implementation of "
                + "the supports."
            )
        solution = sym.solve(equilibirum_equations, unknown_reactions, dict=True)

        # Unpack solution.
        variable_index = 0
        for point_id in reaction_force_points:
            self.points[point_id].reaction_force = solution[0][
                self.points[point_id].reaction_force
            ]
            variable_index = variable_index + 1

        for point_id in reaction_moment_points:
            self.points[point_id].reaction_moment = solution[0][
                self.points[point_id].reaction_moment
            ]
            variable_index = variable_index + 1

    # ----------------------------------------------------------------- solve_internal_loads
    def _solve_internal_loads(self):
        """Finds the shear force and bending moment diagrams for each segment of the beam by
        integrating the differential eqeuilibrium relations.
        """
        # Initialise the internal loads at the starting boundary.
        shear_force_left = -(self.points[0].external_force + self.points[0].reaction_force)
        bending_moment_left = -(
            self.points[0].external_moment + self.points[0].reaction_moment
        )

        # Loop over the segments are find the shear force and bending moment distribution.
        for i in range(len(self.segments)):
            # Shear force.
            # ------------
            C = sym.symbols("C")
            self.segments[i].shear_force = sym.integrate(
                -self.segments[i].distributed_load.expression, x
            )
            sol = sym.solve(
                self.segments[i].shear_force.subs({x: self.segments[i].x_start})
                + C
                - shear_force_left,
                (C),
            )
            C = sol[0]
            self.segments[i].shear_force = self.segments[i].shear_force + C

            # Bending moment
            # --------------
            C = sym.symbols("C")
            self.segments[i].bending_moment = sym.integrate(
                -self.segments[i].shear_force, x
            )
            sol = sym.solve(
                self.segments[i].bending_moment.subs({x: self.segments[i].x_start})
                + C
                - bending_moment_left,
                (C),
            )
            C = sol[0]
            self.segments[i].bending_moment = self.segments[i].bending_moment + C

            # Update the boundary condition for the next segment.
            shear_force_left = (
                self.segments[i].shear_force.subs({x: self.segments[i].x_end})
                - self.points[i + 1].external_force
                - self.points[i + 1].reaction_force
            )
            bending_moment_left = (
                self.segments[i].bending_moment.subs({x: self.segments[i].x_end})
                - self.points[i + 1].external_moment
                - self.points[i + 1].reaction_moment
            )

    # --------------------------------------------------------------------- solve_deflection
    def _solve_deflection(self):
        """Determines the deflection expressions for each segment of the beam by integrating
        the elastic curve equation.
        """
        unknowns_deflection = []
        # Compute the deflection expression at each segment of the beam in terms of the
        # integration coefficients.
        for i in range(len(self.segments)):
            rotation_integration_constant = sym.symbols("S{0}".format(i))
            deflection_integration_constant = sym.symbols("D{0}".format(i))
            unknowns_deflection.append(rotation_integration_constant)
            unknowns_deflection.append(deflection_integration_constant)
            self.segments[i].rotation = (
                sym.integrate(
                    self.segments[i].bending_moment
                    / (self.segments[i].young * self.segments[i].inertia),
                    x,
                )
                + rotation_integration_constant
            )
            self.segments[i].deflection = (
                sym.integrate(self.segments[i].rotation, x)
                + deflection_integration_constant
            )

        # Set up the system of equations with the geometri boundary conditions of the
        # beam and determine the integration coefficients.
        geometry_equations = []
        self.points[0].set_geometric_boundary_conditions(
            [self.segments[0].rotation], [self.segments[0].deflection], geometry_equations
        )

        for i, point in enumerate(self.points[1:-1]):
            self.points[i + 1].set_geometric_boundary_conditions(
                [self.segments[i].rotation, self.segments[i + 1].rotation],
                [self.segments[i].deflection, self.segments[i + 1].deflection],
                geometry_equations,
            )

        self.points[-1].set_geometric_boundary_conditions(
            [self.segments[-1].rotation], [self.segments[-1].deflection], geometry_equations
        )

        sol = sym.solve(geometry_equations, unknowns_deflection, dict=True)
        for i in range(len(self.segments)):
            self.segments[i].rotation = self.segments[i].rotation.subs(
                {unknowns_deflection[2 * i]: sol[0][unknowns_deflection[2 * i]]}
            )
            self.segments[i].deflection = self.segments[i].deflection.subs(
                {unknowns_deflection[2 * i]: sol[0][unknowns_deflection[2 * i]]}
            )
            self.segments[i].deflection = self.segments[i].deflection.subs(
                {unknowns_deflection[2 * i + 1]: sol[0][unknowns_deflection[2 * i + 1]]}
            )

    # -------------------------------------------------------------------------------- solve
    def solve(self, output=True):
        """Solves the beam equilibirum problem, determining the reactions, diagrams of
        internal loads and deflections.
        """
        # Checfk if the properties have been properly set.
        self._check_beam_properties()
        # Set the beam segments with piecewise continuous properties.
        self._set_segments()
        # Solve for the exterior reactions.
        self._solve_reactions()
        # Solver for internal loads.
        self._solve_internal_loads()
        # Solve for deflection.
        self._solve_deflection()

        # Output the results.
        if output:
            self._print_points()
            self._print_segments()
            self._print_reactions()
            self._print_internal_loads()
            self._print_deflections()

    # --------------------------------------------------------------------------------- plot
    def plot(self, subs={}):
        """Plots the shear force and bending moment diagrams and the deflection.

        Parameters
        ----------
        subs : dictionary
          User-specified symbols substitution for the symbolic expressions

        Returns
        -------
        fig : matplotlib figure
          Plotting figure
        ax : list of matplotlib axis
          List of subplot axis
        """
        # Define some plotting settings.
        color_bending_moment = "firebrick"
        color_shear_force = "forestgreen"
        color_deflection = "black"
        color_distributed_load = "royalblue"
        color_beam = "black"
        line_width_distributed_loads = 2
        line_width_diagrams = 2
        line_width_deflection = 3
        line_width_beam = 3
        alpha = 0.35
        max_distributed_load = []
        xmin = 0
        xmax = 0

        # Remove the 'x' variable from the user substitutions
        subs.pop("x", None)

        # Create the figure and plot the shear force, bending moment and deflection for
        # each segment.
        fig = plt.figure(num="Internal loads and deflection", figsize=(7, 8))
        gs = fig.add_gridspec(
            nrows=4,
            ncols=1,
            left=0.12,
            right=0.95,
            bottom=0.07,
            top=0.95,
            hspace=0.10,
            width_ratios=[1],
            height_ratios=[1, 1, 1, 1],
        )
        ax = [fig.add_subplot(x) for x in gs]

        # Plots segments
        # --------------
        for i, isegment in enumerate(self.segments):
            # Copies of the relevant expressions
            distributed_load_plot = isegment.distributed_load.expression
            shear_force_plot = isegment.shear_force
            bending_moment_plot = isegment.bending_moment
            deflection_plot = isegment.deflection
            x_start_plot = isegment.x_start
            x_end_plot = isegment.x_end

            # User defined substitutions
            distributed_load_plot = distributed_load_plot.subs(subs)
            shear_force_plot = shear_force_plot.subs(subs)
            bending_moment_plot = bending_moment_plot.subs(subs)
            deflection_plot = deflection_plot.subs(subs)
            x_start_plot = x_start_plot.subs(subs)
            x_end_plot = x_end_plot.subs(subs)

            # Create new expressions by substituting all remaining symbolic variables with
            # '1', except for the x variable
            variables_distributed_load = distributed_load_plot.free_symbols
            variables_distributed_load.discard(x)
            variables_shear_force = shear_force_plot.free_symbols
            variables_shear_force.discard(x)
            variables_bending_moment = bending_moment_plot.free_symbols
            variables_bending_moment.discard(x)
            variables_deflection = deflection_plot.free_symbols
            variables_deflection.discard(x)
            variables_x_start = x_start_plot.free_symbols
            variables_x_start.discard(x)
            variables_x_end = x_end_plot.free_symbols
            variables_x_end.discard(x)

            for ivariable in variables_distributed_load:
                distributed_load_plot = distributed_load_plot.subs({ivariable: 1})

            for ivariable in variables_shear_force:
                shear_force_plot = shear_force_plot.subs({ivariable: 1})

            for ivariable in variables_bending_moment:
                bending_moment_plot = bending_moment_plot.subs({ivariable: 1})

            for ivariable in variables_deflection:
                deflection_plot = deflection_plot.subs({ivariable: 1})

            for ivariable in variables_x_start:
                x_start_plot = x_start_plot.subs({ivariable: 1})

            for ivariable in variables_x_end:
                x_end_plot = x_end_plot.subs({ivariable: 1})

            # Numeric plotting x variable.
            x_plot = np.linspace(
                float(x_start_plot), float(x_end_plot), num=100, endpoint=True
            )

            # Distributed load plot.
            distributed_load_numeric = np.vectorize(sym.lambdify(x, distributed_load_plot))(
                x_plot
            )
            max_distributed_load.append(np.max(np.abs(distributed_load_numeric)))
            ax[0].plot(
                x_plot,
                -distributed_load_numeric,
                color=color_distributed_load,
                linewidth=line_width_distributed_loads,
            )
            ax[0].fill_between(
                x_plot,
                -distributed_load_numeric,
                color=color_distributed_load,
                linewidth=line_width_distributed_loads,
                alpha=alpha,
            )
            ax[0].plot(
                [x_plot[0], x_plot[-1]], [0, 0], color=color_beam, linewidth=line_width_beam
            )

            # Shear force plot.
            ax[1].plot(
                x_plot,
                np.vectorize(sym.lambdify(x, shear_force_plot))(x_plot),
                color=color_shear_force,
                linewidth=line_width_diagrams,
            )
            ax[1].fill_between(
                x_plot,
                sym.lambdify(x, shear_force_plot)(x_plot),
                color=color_shear_force,
                alpha=alpha,
            )

            # Bending diagram plot.
            ax[2].plot(
                x_plot,
                np.vectorize(sym.lambdify(x, bending_moment_plot))(x_plot),
                color=color_bending_moment,
                linewidth=line_width_diagrams,
            )
            ax[2].fill_between(
                x_plot,
                sym.lambdify(x, bending_moment_plot)(x_plot),
                color=color_bending_moment,
                alpha=alpha,
            )

            # Deflection plot.
            ax[3].plot(
                x_plot,
                np.vectorize(sym.lambdify(x, deflection_plot))(x_plot),
                color=color_deflection,
                linewidth=line_width_deflection,
            )
            ax[3].plot(
                [x_plot[0], x_plot[-1]],
                [0, 0],
                color=color_beam,
                linewidth=line_width_beam / 2,
                linestyle="--",
            )

            # Get maximum and minimum coordinate of the beam (axis limits).
            if i == 0:
                xmin = x_plot[0]
            if i == len(self.segments) - 1:
                xmax = x_plot[-1]

        # Set the y-axis upper and lower bounds for the beam representation.
        ymax = max(max_distributed_load) * 1.1
        if ymax < tol:
            ymax = 1.0

        # Fix the limits of the first axis. This is mandatory for plotting the beam
        # configuration.
        ymin = -ymax
        ax[0].set_ylim(ymin, ymax)
        ax[0].set_xlim(xmin, xmax)
        ax[0].axis("off")

        # Plot points
        # -----------
        external_force_plot_vector = np.zeros((len(self.points)))
        external_moment_plot_vector = np.zeros((len(self.points)))
        # Loop over the points, draw the points and extract the magnitudes of the external
        # forces and moments.
        for i, ipoint in enumerate(self.points):
            ipoint.draw_support(ax[0], input_substitution=subs)
            external_force_plot = ipoint.external_force
            external_moment_plot = ipoint.external_moment

            # User-defined substitutions
            external_force_plot = external_force_plot.subs(subs)
            external_moment_plot = external_moment_plot.subs(subs)

            for ivariable in external_force_plot.free_symbols:
                external_force_plot = external_force_plot.subs({ivariable: 1})

            for ivariable in external_moment_plot.free_symbols:
                external_moment_plot = external_moment_plot.subs({ivariable: 1})

            external_force_plot_vector[i] = float(external_force_plot)
            external_moment_plot_vector[i] = float(external_moment_plot)

        # Scale the forces so that the maximum point load has the length of the upper bound
        # of the configuration plot and draw the forces.
        max_force = np.max(np.abs(external_force_plot_vector))
        max_moment = np.max(np.abs(external_moment_plot_vector))
        fraction_max = 0.9
        if max_force > tol:
            external_force_plot_vector = (
                external_force_plot_vector / max_force * ymax * fraction_max
            )

        if max_moment > tol:
            external_moment_plot_vector = external_moment_plot_vector / max_moment

        for i, ipoint in enumerate(self.points):
            if abs(external_force_plot_vector[i]) > tol:
                ipoint.draw_force(
                    ax[0], external_force_plot_vector[i], input_substitution=subs
                )
            if abs(external_moment_plot_vector[i]) > tol:
                ipoint.draw_moment(
                    ax[0], external_moment_plot_vector[i], input_substitution=subs
                )

        # Axis labels.
        ax[1].set_ylabel(r"Shear force, $V(x)$")
        ax[2].set_ylabel(r"Bending moment, $M(x)$")
        ax[3].set_ylabel(r"Deflection, $v(x)$")
        ax[3].set_xlabel(r"Coordinate, $x$")

        # Axis ticks format.
        ax[1].yaxis.set_major_formatter(ticker.FormatStrFormatter("%0.0e"))
        ax[1].xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.0e"))
        ax[2].yaxis.set_major_formatter(ticker.FormatStrFormatter("%0.0e"))
        ax[2].xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.0e"))
        ax[3].yaxis.set_major_formatter(ticker.FormatStrFormatter("%0.0e"))
        ax[3].xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.0e"))

        # Set the x-axis settings
        ax[1].set_xlim(xmin, xmax)
        ax[1].set_xticklabels([])
        ax[2].set_xlim(xmin, xmax)
        ax[2].set_xticklabels([])
        ax[3].set_xlim(xmin, xmax)

        return fig, ax
    
    # -------------------------------------------------------------------------------- solve
    def solve_v3(self, subs={}, output=True):
        """Solves the beam equilibrium problem and generates numerical data for plotting.
        
        This method determines the reactions, internal loads, and deflections, then
        generates numerical data for plotting the shear force, bending moment diagrams 
        and deflection.

        Parameters
        ----------
        subs : dictionary
          User-specified symbols substitution for the symbolic expressions
        output : bool
          Whether to print the results to console

        Returns
        -------
        plot_data : dict
          Dictionary containing numerical data for plotting with keys:
          - 'points': beam points information
          - 'segments': beam segments information  
          - 'reactions': reaction forces and moments
          - 'internal_loads': shear force and bending moment expressions
          - 'deflections': rotation and deflection expressions
          - 'x_coord': list of x-coordinates
          - 'shear_force': list of shear force values
          - 'bending_moment': list of bending moment values
          - 'slope': list of slope/rotation values
          - 'deflection': list of deflection values
          - 'shear_force_info': dictionary with local extrema and null points
          - 'bending_moment_info': dictionary with local extrema and null points
          - 'slope_info': dictionary with local extrema and null points
          - 'deflection_info': dictionary with local extrema and null points
          - 'symbolic_info': dictionary with slope_null (exact symbolic solutions where slope equals 0)
            and deflection_minimum (symbolic minimum deflection location and value)
        """
        # Solve the beam equilibrium problem
        self._check_beam_properties()
        self._set_segments()
        self._solve_reactions()
        self._solve_internal_loads()
        self._solve_deflection()

        # Output the results if requested
        if output:
            self._print_points()
            self._print_segments()
            self._print_reactions()
            self._print_internal_loads()
            self._print_deflections()

        # Create base beam data dictionary
        beam_data = {
            'points': self._save_points(),
            'segments': self._save_segments(), 
            'reactions': self._save_reactions(),
            'internal_loads': self._save_internal_loads(),
            'deflections': self._save_deflections()
        }
        
        # Generate numerical plotting data
        plot_data = self._generate_plot_data(subs)
        
        # Combine beam data with plot data
        plot_data.update(beam_data)
        
        # Save the plot data to JSON
        # with open('plot_data.json', 'w') as f:
        #     json.dump(plot_data, f, indent=4)

        return plot_data

    def _generate_plot_data(self, subs):
        """Generates numerical data for plotting the beam diagrams.
        
        Parameters
        ----------
        subs : dictionary
          User-specified symbols substitution for the symbolic expressions

        Returns
        -------
        plot_data : dict
          Dictionary containing numerical plotting data
        """
        # Remove the 'x' variable from the user substitutions
        subs.pop("x", None)

        # Initialize data collection lists
        all_x_coords = []
        all_shear_forces = []
        all_bending_moments = []
        all_slopes = []
        all_deflections = []
        all_symbolic_slope_zeros = []  # Collect symbolic zero locations

        # Process segments to generate numerical data
        for isegment in self.segments:
            # Get expressions and apply substitutions
            expressions = self._prepare_segment_expressions(isegment, subs)
            
            # Generate x-coordinates for this segment
            x_plot = np.linspace(
                float(expressions['x_start']), 
                float(expressions['x_end']), 
                num=100, 
                endpoint=True
            )

            # Collect numerical data for this segment
            all_x_coords.extend(x_plot.tolist())
            
            # Shear force data
            shear_force_numeric = np.vectorize(sym.lambdify(x, expressions['shear_force']))(x_plot)
            all_shear_forces.extend(shear_force_numeric.tolist())
            
            # Bending moment data
            bending_moment_numeric = np.vectorize(sym.lambdify(x, expressions['bending_moment']))(x_plot)
            all_bending_moments.extend(bending_moment_numeric.tolist())
            
            # Slope data (rotation)
            slope_numeric = np.vectorize(sym.lambdify(x, expressions['slope']))(x_plot)
            all_slopes.extend(slope_numeric.tolist())
            
            # Deflection data
            deflection_numeric = np.vectorize(sym.lambdify(x, expressions['deflection']))(x_plot)
            all_deflections.extend(deflection_numeric.tolist())
            
            # Find symbolic zeros for slope (where slope equals 0)
            symbolic_zeros = self._find_where_slope_equals_zero(isegment, subs)
            # Format as [x, 0] pairs to match the format of 'null' results
            for x_zero in symbolic_zeros:
                all_symbolic_slope_zeros.append([x_zero, 0.0])

        # Create the plot data dictionary
        plot_data = {
            'x_coord': all_x_coords,
            'shear_force': all_shear_forces,
            'bending_moment': all_bending_moments,
            'slope': all_slopes,
            'deflection': all_deflections
        }

        # Convert to numpy arrays for analysis
        x_coords_array = np.array(all_x_coords)
        shear_forces_array = np.array(all_shear_forces)
        bending_moments_array = np.array(all_bending_moments)
        slopes_array = np.array(all_slopes)
        deflections_array = np.array(all_deflections)

        # Add analysis information
        plot_data['shear_force_info'] = {
            'local_extrema': self._find_local_extrema(x_coords_array, shear_forces_array),
            'null': self._find_null_values_and_locations(x_coords_array, shear_forces_array)
        }
        
        plot_data['bending_moment_info'] = {
            'local_extrema': self._find_local_extrema(x_coords_array, bending_moments_array),
            'null': self._find_null_values_and_locations(x_coords_array, bending_moments_array)
        }
        
        plot_data['slope_info'] = {
            'local_extrema': self._find_local_extrema(x_coords_array, slopes_array),
            'null': self._find_null_values_and_locations(x_coords_array, slopes_array)
        }
        
        plot_data['deflection_info'] = {
            'local_extrema': self._find_local_extrema(x_coords_array, deflections_array),
            'null': self._find_null_values_and_locations(x_coords_array, deflections_array)
        }

        plot_data['symbolic_info'] = {
            'slope_null': all_symbolic_slope_zeros,
            'deflection_minimum': self._find_minimum_deflection(subs)
        }

        return plot_data

    def _prepare_segment_expressions(self, segment, subs):
        """Prepares segment expressions for numerical evaluation.
        
        Parameters
        ----------
        segment : _segment
          The beam segment to process
        subs : dict
          Symbol substitutions to apply

        Returns
        -------
        expressions : dict
          Dictionary containing prepared expressions
        """
        # Get expressions and apply user substitutions
        expressions = {
            'distributed_load': segment.distributed_load.expression.subs(subs),
            'shear_force': segment.shear_force.subs(subs),
            'bending_moment': segment.bending_moment.subs(subs),
            'deflection': segment.deflection.subs(subs),
            'slope': segment.rotation.subs(subs),
            'x_start': segment.x_start.subs(subs),
            'x_end': segment.x_end.subs(subs)
        }
        
        # Substitute remaining symbolic variables with 1 (except x)
        for key in expressions:
            if key not in ['x_start', 'x_end']:
                variables = expressions[key].free_symbols
                variables.discard(x)
                for variable in variables:
                    expressions[key] = expressions[key].subs({variable: 1})
            else:
                variables = expressions[key].free_symbols
                variables.discard(x)
                for variable in variables:
                    expressions[key] = expressions[key].subs({variable: 1})
        
        return expressions

    def _find_where_slope_equals_zero(self, segment, subs):
        """Finds where the slope (rotation) equals zero symbolically.
        
        Parameters
        ----------
        segment : _segment
          The beam segment to process
        subs : dict
          Symbol substitutions to apply
        
        Returns
        -------
        zero_locations : list
          List of x-coordinates where slope equals zero (within segment bounds)
        """
        # Get the slope expression with substitutions applied
        slope_expr = segment.rotation.subs(subs)
        
        # Apply remaining substitutions (set other variables to 1, keep x)
        variables = slope_expr.free_symbols
        variables.discard(x)
        for variable in variables:
            slope_expr = slope_expr.subs({variable: 1})
        
        # Get segment bounds
        x_start = float(segment.x_start.subs(subs))
        x_end = float(segment.x_end.subs(subs))
        
        # Solve slope == 0 for x
        try:
            solutions = sym.solve(slope_expr, x, dict=True)
        except:
            # If solve fails, return empty list
            return []
        
        zero_locations = []
        
        # Handle both dict=True format (list of dicts) and dict=False format (list of expressions)
        if solutions and isinstance(solutions[0], dict):
            # Extract x values from dictionary solutions
            for sol_dict in solutions:
                if x in sol_dict:
                    sol = sol_dict[x]
                    try:
                        # Check if solution is real
                        if sol.is_real or sol.is_real is None:
                            x_val = float(sol)
                            # Check if within segment bounds
                            if x_start <= x_val <= x_end:
                                zero_locations.append(x_val)
                    except (TypeError, ValueError):
                        # Skip non-numeric solutions
                        continue
        else:
            # Handle list of expressions
            for sol in solutions:
                try:
                    # Check if solution is real
                    if sol.is_real or sol.is_real is None:
                        x_val = float(sol)
                        # Check if within segment bounds
                        if x_start <= x_val <= x_end:
                            zero_locations.append(x_val)
                except (TypeError, ValueError):
                    # Skip non-numeric solutions
                    continue
        
        # Remove duplicates and sort
        zero_locations = sorted(list(set(zero_locations)))
        
        return zero_locations

    def _find_minimum_deflection(self, subs):
        """Finds the symbolic minimum deflection across all beam segments.
        
        The minimum deflection occurs at points where:
        1. The slope (rotation) equals zero (deflection extrema)
        2. Segment endpoints
        
        Parameters
        ----------
        subs : dict
          Symbol substitutions to apply
        
        Returns
        -------
        min_deflection : dict
          Dictionary with keys:
          - 'x_location': x-coordinate where minimum deflection occurs
          - 'deflection_value': minimum deflection value (as SymPy expression or float)
          - 'segment_index': index of segment containing minimum deflection
        """
        min_deflection_value = None
        min_deflection_x = None
        min_segment_idx = None
        
        # Process each segment
        for seg_idx, segment in enumerate(self.segments):
            # Get deflection expression with substitutions
            deflection_expr = segment.deflection.subs(subs)
            
            # Apply remaining substitutions (set other variables to 1, keep x)
            variables = deflection_expr.free_symbols
            variables.discard(x)
            for variable in variables:
                deflection_expr = deflection_expr.subs({variable: 1})
            
            # Get segment bounds
            x_start = float(segment.x_start.subs(subs))
            x_end = float(segment.x_end.subs(subs))
            
            # Candidate points: endpoints and where slope = 0
            candidate_x = [x_start, x_end]
            
            # Find where slope equals zero (deflection extrema)
            slope_zeros = self._find_where_slope_equals_zero(segment, subs)
            candidate_x.extend(slope_zeros)
            
            # Remove duplicates
            candidate_x = sorted(list(set(candidate_x)))
            
            # Evaluate deflection at each candidate point
            for x_candidate in candidate_x:
                try:
                    # Evaluate deflection at this x value
                    deflection_at_x = deflection_expr.subs({x: x_candidate})
                    
                    # Convert to float if possible
                    try:
                        deflection_float = float(deflection_at_x)
                    except (TypeError, ValueError):
                        # If it's still symbolic, try to simplify and evaluate
                        deflection_float = float(sym.N(deflection_at_x))
                    
                    # Check if this is the minimum so far
                    if min_deflection_value is None or deflection_float < min_deflection_value:
                        min_deflection_value = deflection_float
                        min_deflection_x = x_candidate
                        min_segment_idx = seg_idx
                        
                except (TypeError, ValueError, AttributeError):
                    # Skip if evaluation fails
                    continue
        
        if min_deflection_value is None:
            return {
                'x_location': None,
                'deflection_value': None,
                'segment_index': None
            }
        
        return {
            'x_location': min_deflection_x,
            'deflection_value': min_deflection_value,
            'segment_index': min_segment_idx
        }

    def _find_local_extrema(self, x_coords, values):
        """Finds all local extrema (local maxima and local minima) in the data.
        
        Parameters
        ----------
        x_coords : numpy.ndarray
          Array of x-coordinates
        values : numpy.ndarray
          Array of function values

        Returns
        -------
        extrema : list
          List of [x_coord, value] pairs for local extrema
        """
        extrema = []
        candidate_indices = []
        
        # Check interior points using bandwidth of 5
        if len(values) >= 5:
            for i in range(2, len(values) - 2):
                left_side = [values[i-2], values[i-1]]
                right_side = [values[i+1], values[i+2]]
                
                # Local maximum: middle value is higher than at least one point on each side
                if (values[i] > min(left_side) and values[i] > min(right_side)):
                    candidate_indices.append(i)
                # Local minimum: middle value is lower than at least one point on each side
                elif (values[i] < max(left_side) and values[i] < max(right_side)):
                    candidate_indices.append(i)
        
        # Handle cases with fewer than 5 points using 3-point method
        else:
            for i in range(1, len(values) - 1):
                # Local maximum: value is greater than both neighbors
                if values[i] > values[i-1] and values[i] > values[i+1]:
                    candidate_indices.append(i)
                # Local minimum: value is less than both neighbors
                elif values[i] < values[i-1] and values[i] < values[i+1]:
                    candidate_indices.append(i)
        
        # Check endpoints for extrema
        if len(values) > 1:
            # First point
            if values[0] > values[1] or values[0] < values[1]:
                candidate_indices.append(0)
            
            # Last point
            if values[-1] > values[-2] or values[-1] < values[-2]:
                candidate_indices.append(len(values) - 1)
        
        # Filter out adjacent extrema - keep only one if two are adjacent
        if candidate_indices:
            candidate_indices.sort()
            i = 0
            while i < len(candidate_indices):
                current_idx = candidate_indices[i]
                extrema.append([x_coords[current_idx], values[current_idx]])
                
                # Skip the next candidate if it's adjacent to the current one
                if i + 1 < len(candidate_indices) and candidate_indices[i + 1] == current_idx + 1:
                    i += 2  # Skip both current and next
                else:
                    i += 1  # Move to next candidate
        
        return extrema

    def _find_null_values_and_locations(self, x_coords, values):
        """Finds null values (zero crossings and zero values) in the data.
        
        Parameters
        ----------
        x_coords : numpy.ndarray
          Array of x-coordinates
        values : numpy.ndarray
          Array of function values

        Returns
        -------
        null_locations : list
          List of [x_coord, 0] pairs for zero crossings and zero values
        """
        null_locations = []
        
        # Helper function to check if a point is already in the list
        def is_duplicate(x_coord, tolerance=1e-6):
            for existing_point in null_locations:
                if abs(existing_point[0] - x_coord) < tolerance:
                    return True
            return False
        
        # Find exact zero values
        zero_indices = np.where(values == 0)[0]
        for i in zero_indices:
            if not is_duplicate(x_coords[i]):
                null_locations.append([x_coords[i], 0])
        
        # Find zero crossings (sign changes)
        for i in range(len(values) - 1):
            if (values[i] < 0 and values[i + 1] > 0) or (values[i] > 0 and values[i + 1] < 0):
                # Linear interpolation to find approximate x-coordinate of zero crossing
                x1, x2 = x_coords[i], x_coords[i + 1]
                y1, y2 = values[i], values[i + 1]
                
                # Avoid division by zero
                if y2 != y1:
                    x_zero = x1 - y1 * (x2 - x1) / (y2 - y1)
                    if not is_duplicate(x_zero):
                        null_locations.append([x_zero, 0])
        
        # Check endpoints for near-zero values (less than 0.0001)
        if len(values) > 0:
            # First point
            if abs(values[0]) < 0.0001:
                if not is_duplicate(x_coords[0]):
                    null_locations.append([x_coords[0], 0])
            # Last point
            if abs(values[-1]) < 0.0001:
                if not is_duplicate(x_coords[-1]):
                    null_locations.append([x_coords[-1], 0])
        
        return null_locations

    # ------------------------------------------------------------------------- print_points
    def _print_points(self):
        """Prints the information of points identified along the beam."""
        print("\n{0:^83}".format("Beam points"))
        print(83 * "=")
        print(
            "{0:^20} {1:^20} {2:^20} {3:^20}".format("Coordinate", "Type", "Load", "Moment")
        )
        print(83 * "-")
        for ipoint in self.points:
            x_coord_str = self._trim_trailing_zeros(ipoint.x_coord)
            print(
                "{0:^20} {1:^20} {2:^20} {3:^20}".format(
                    x_coord_str,
                    ipoint.get_name(),
                    str(ipoint.external_force),
                    str(ipoint.external_moment),
                )
            )

        print(83 * "=" + "\n")

    # ----------------------------------------------------------------------- print_segments
    def _print_segments(self):
        """Prints the information of the identified segments."""
        print("\n{0:^83}".format("Beam segments"))
        print(83 * "=")
        print(
            "{0:^20} {1:^20} {2:^20} {3:^20}".format(
                "Span", "Young modulus", "Inertia", "Distributed load"
            )
        )
        print(83 * "-")
        for isegment in self.segments:
            # Trim decimal places when numeric
            x_start_str = self._trim_trailing_zeros(isegment.x_start)
            x_end_str = self._trim_trailing_zeros(isegment.x_end)
            span_string = "[ {0:^5} - {1:^5} ]".format(x_start_str, x_end_str)
            print(
                "{0:^20} {1:^20} {2:^20} {3:^20}".format(
                    span_string,
                    str(isegment.young),
                    str(isegment.inertia),
                    str(isegment.distributed_load.expression),
                )
            )

        print(83 * "=" + "\n")

    # ---------------------------------------------------------------------- print_reactions
    def _print_reactions(self):
        """Prints the reactions forces."""
        print("\n{0:^83}".format("Exterior Reactions"))
        print(83 * "=")
        print("{0:^27} {1:^27} {2:^27}".format("Point", "Type", "Value"))
        print(83 * "-")
        for ipoint in self.points:
            if ipoint.has_reaction_force():
                x_coord_str = self._trim_trailing_zeros(ipoint.x_coord)
                print(
                    "{0:^27} {1:^27} {2:^27}".format(
                        x_coord_str, "Force", str(ipoint.reaction_force)
                    )
                )

            if ipoint.has_reaction_moment():
                x_coord_str = self._trim_trailing_zeros(ipoint.x_coord)
                print(
                    "{0:^27} {1:^27} {2:^27}".format(
                        x_coord_str, "Moment", str(ipoint.reaction_moment)
                    )
                )

        print(83 * "=" + "\n")

    # ----------------------------------------------------------------- print_internal_loads
    def _print_internal_loads(self):
        """Prints the shear force and bending moment expression for each segment."""
        print("\n{0:^83}".format("Internal Loads"))
        print(83 * "=")
        print("{0:^20} {1:^10} {2:^50}".format("Span", "Diagram", "Expression"))
        for isegment in self.segments:
            x_start_str = self._trim_trailing_zeros(isegment.x_start)
            x_end_str = self._trim_trailing_zeros(isegment.x_end)
            print(83 * "-")
            span_string = "[ {0:^5} - {1:^5} ]".format(x_start_str, x_end_str)
            print(
                "{0:^20} {1:^10} {2:^50}".format(
                    span_string, "V(x)", str(isegment.shear_force)
                )
            )
            print(
                "{0:^20} {1:^10} {2:^50}".format(
                    span_string, "M(x)", str(isegment.bending_moment)
                )
            )

        print(83 * "=" + "\n")

    # -------------------------------------------------------------------- print_deflections
    def _print_deflections(self):
        """Prints the shear force and bending moment expression for each segment."""
        print("\n{0:^83}".format("Rotation and deflection"))
        print(83 * "=")
        print("{0:^20} {1:^10} {2:^50}".format("Span", "Variable", "Expression"))
        for isegment in self.segments:
            x_start_str = self._trim_trailing_zeros(isegment.x_start)
            x_end_str = self._trim_trailing_zeros(isegment.x_end)
            print(83 * "-")
            span_string = "[ {0:^5} - {1:^5} ]".format(x_start_str, x_end_str)
            print(
                "{0:^20} {1:^10} {2:^50}".format(
                    span_string, "v(x)", str(isegment.deflection)
                )
            )
            print(
                "{0:^20} {1:^10} {2:^50}".format(
                    span_string, "dv/dx(x)", str(isegment.rotation)
                )
            )

        print(83 * "=" + "\n")

    # --------------------------------------------------------------------------------- save

    def _save_points(self):
        """Returns the information of points identified along the beam as a dictionary.
        
        Returns
        -------
        points_dict : dict
            Dictionary containing points information with keys:
            - 'header': str - Title of the data
            - 'points': list - List of dictionaries, each containing:
                - 'coordinate': str - x-coordinate of the point
                - 'type': str - Type of the point (support type or continuity)
                - 'load': str - External force at the point
                - 'moment': str - External moment at the point
        """
        points_data = {
            'header': 'Beam points',
            'points': []
        }
        
        for ipoint in self.points:
            x_coord_str = self._trim_trailing_zeros(ipoint.x_coord)
            point_info = {
                'coordinate': x_coord_str,
                'type': ipoint.get_name(),
                'load': str(ipoint.external_force),
                'moment': str(ipoint.external_moment)
            }
            points_data['points'].append(point_info)
        
        return points_data

    def _save_segments(self):
        """Returns the information of the identified segments as a dictionary.
        
        Returns
        -------
        segments_dict : dict
            Dictionary containing segments information with keys:
            - 'header': str - Title of the data
            - 'segments': list - List of dictionaries, each containing:
                - 'span': list - List containing [x_start, x_end] coordinates
                - 'young_modulus': str - Young's modulus for the segment
                - 'inertia': str - Moment of inertia for the segment
                - 'distributed_load': str - Distributed load expression
        """
        segments_data = {
            'header': 'Beam segments',
            'segments': []
        }
        
        for isegment in self.segments:
            # Trim decimal places when numeric
            x_start_str = self._trim_trailing_zeros(isegment.x_start)
            x_end_str = self._trim_trailing_zeros(isegment.x_end)
            
            segment_info = {
                'span': [x_start_str, x_end_str],
                'young_modulus': str(isegment.young),
                'inertia': str(isegment.inertia),
                'distributed_load': str(isegment.distributed_load.expression)
            }
            segments_data['segments'].append(segment_info)
        
        return segments_data

    def _save_reactions(self):
        """Returns the reactions forces as a dictionary.
        
        Returns
        -------
        reactions_dict : dict
            Dictionary containing reactions information with keys:
            - 'header': str - Title of the data
            - 'reactions': list - List of dictionaries, each containing:
                - 'point': str - x-coordinate of the reaction point
                - 'type': str - Type of reaction (Force or Moment)
                - 'value': str - Reaction value
        """
        reactions_data = {
            'header': 'Exterior Reactions',
            'reactions': []
        }
        
        for ipoint in self.points:
            if ipoint.has_reaction_force():
                x_coord_str = self._trim_trailing_zeros(ipoint.x_coord)
                reaction_info = {
                    'point': x_coord_str,
                    'type': 'Force',
                    'value': str(ipoint.reaction_force)
                }
                reactions_data['reactions'].append(reaction_info)

            if ipoint.has_reaction_moment():
                x_coord_str = self._trim_trailing_zeros(ipoint.x_coord)
                reaction_info = {
                    'point': x_coord_str,
                    'type': 'Moment',
                    'value': str(ipoint.reaction_moment)
                }
                reactions_data['reactions'].append(reaction_info)
        
        return reactions_data

    def _save_internal_loads(self):
        """Returns the shear force and bending moment expressions as a dictionary.
        
        Returns
        -------
        internal_loads_dict : dict
            Dictionary containing internal loads information with keys:
            - 'header': str - Title of the data
            - 'segments': list - List of dictionaries, each containing:
                - 'span': str - Span range [x_start - x_end]
                - 'shear_force': str - Shear force expression V(x)
                - 'bending_moment': str - Bending moment expression M(x)
        """
        internal_loads_data = {
            'header': 'Internal Loads',
            'segments': []
        }
        
        for isegment in self.segments:
            x_start_str = self._trim_trailing_zeros(isegment.x_start)
            x_end_str = self._trim_trailing_zeros(isegment.x_end)
            
            segment_info = {
                'span': [x_start_str, x_end_str],
                'shear_force': str(isegment.shear_force),
                'bending_moment': str(isegment.bending_moment)
            }
            internal_loads_data['segments'].append(segment_info)
        
        return internal_loads_data

    def _save_deflections(self):
        """Returns the rotation and deflection expressions as a dictionary.
        
        Returns
        -------
        deflections_dict : dict
            Dictionary containing deflection information with keys:
            - 'header': str - Title of the data
            - 'segments': list - List of dictionaries, each containing:
                - 'span': str - Span range [x_start - x_end]
                - 'deflection': str - Deflection expression v(x)
                - 'rotation': str - Rotation expression dv/dx(x)
        """
        deflections_data = {
            'header': 'Rotation and deflection',
            'segments': []
        }
        
        for isegment in self.segments:
            x_start_str = self._trim_trailing_zeros(isegment.x_start)
            x_end_str = self._trim_trailing_zeros(isegment.x_end)
            
            segment_info = {
                'span': [x_start_str, x_end_str],
                'deflection': str(isegment.deflection),
                'rotation': str(isegment.rotation)
            }
            deflections_data['segments'].append(segment_info)
        
        return deflections_data

    # --------------------------------------------------------------------------------- save
    def save(self, file_name):
        """Saves the beam information to a JSON file."""
        with open(file_name, 'w') as f:
            json.dump(self._save_points(), f, indent=4)
            json.dump(self._save_segments(), f, indent=4)
            json.dump(self._save_reactions(), f, indent=4)
            json.dump(self._save_internal_loads(), f, indent=4)
            json.dump(self._save_deflections(), f, indent=4)

    # ------------------------------------------------------------------ trim_trailing_zeros
    @staticmethod
    def _trim_trailing_zeros(expr):
        """Removes the trailing zeros from a SymPy expression containing only numbers.

        Parameters
        ----------
        expr : SymPy expression
          Input expression

        Returns
        -------
        expr_trimmed : SymPy expression
          Output (trimmed expression)
        """
        expr_trimmed = str(expr)
        if len(expr.free_symbols) == 0 and "." in expr_trimmed:
            expr_trimmed = expr_trimmed.rstrip("0")
            if expr_trimmed[-1] == ".":
                expr_trimmed = expr_trimmed[:-1]

        return expr_trimmed


# ========================================================================= property_segment
class _property_segment:
    """Class for segments properties in a symbolic-compatible fashion."""

    def __init__(self, x_start, x_end, value):
        self.x_start = sym.sympify(x_start)
        self.x_end = sym.sympify(x_end)
        self.value = sym.sympify(value)


# ================================================================================== segment
class _segment:
    """Beam segments with locally continuous properties and loadings."""

    def __init__(self, x_start, x_end, distributed_load, young, inertia):
        self.x_start = sym.sympify(x_start)
        self.x_end = sym.sympify(x_end)
        self.distributed_load = distributed_load
        self.young = sym.sympify(young)
        self.inertia = sym.sympify(inertia)

        # Iniitialise the expressions of the bending moment and shear force diagrams.
        self.shear_force = sym.sympify(0)
        self.bending_moment = sym.sympify(0)
        self.rotation = sym.sympify(0)
        self.deflection = sym.sympify(0)


# ==========================================================================================