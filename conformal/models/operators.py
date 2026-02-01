from itertools import product
import torch
import ltn
from conformal.general_utils import log

# ===============
# MNIST
# ===============


def mnist_circuit(sequence_len=2, n_digits=10, oputput_dim=19):
    possible_worlds = list(product(range(n_digits), repeat=sequence_len))
    n_worlds = len(possible_worlds)
    n_queries = len(range(0, oputput_dim))
    look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}
    w_q = torch.zeros(n_worlds, n_queries)  # (100, 20)
    for w in range(n_worlds):
        digit1, digit2 = look_up[w]
        for q in range(n_queries):
            if digit1 + digit2 == q:
                w_q[w, q] = 1
    return w_q


def mnist_sump_circuit(sequence_len=2, n_digits=10, oputput_dim=2):
    possible_worlds = list(product(range(n_digits), repeat=sequence_len))
    n_worlds = len(possible_worlds)
    n_queries = len(range(0, oputput_dim))
    look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}
    w_q = torch.zeros(n_worlds, n_queries)  # (100, 2)
    for w in range(n_worlds):
        digit1, digit2 = look_up[w]
        for q in range(n_queries):
            if (digit1 + digit2) % 2 == q:
                w_q[w, q] = 1
    return w_q


class BaseMNISTLTNLoss(torch.nn.Module):
    """
    Base class for LTN-based MNIST logic losses.
    Encapsulates the common logic for variable creation and satisfaction aggregation.
    """

    def __init__(self, and_op, exists_op, forall_op, n_outputs) -> None:
        super().__init__()
        self.and_op = and_op
        self.exists_op = exists_op
        self.forall_op = forall_op
        self.n_outputs = n_outputs

    def condition(self):
        """Override this method to define the specific logic rule."""
        raise NotImplementedError

    def forward(self, pred_concepts, labels):
        # Variables representing the two input images and the target label
        x = ltn.Variable("x", pred_concepts[:, 0])
        y = ltn.Variable("y", pred_concepts[:, 1])
        n = ltn.Variable("n", labels)

        # LTN predicate for digit classification
        digit_pred = ltn.Predicate(
            func=lambda digits, d_idx: torch.gather(digits, 1, d_idx)
        )

        # Range variables for all possible digit values (0-9)
        d1 = ltn.Variable("d1", torch.arange(pred_concepts.shape[-1]))
        d2 = ltn.Variable("d2", torch.arange(pred_concepts.shape[-1]))

        # The core logical formula: Forall x,y,n: Exists d1,d2 such that (d1+d2 satisfy condition)
        sat_agg = self.forall_op(
            ltn.diag(x, y, n),
            self.exists_op(
                [d1, d2],
                self.and_op(digit_pred(x, d1), digit_pred(y, d2)),
                cond_vars=[d1, d2, n],
                cond_fn=self.condition(),
            ),
        )

        log(f"LTN loss: {1 - sat_agg.value}", "DEBUG")

        return 1 - sat_agg.value


class mnist_add_ltn_loss(BaseMNISTLTNLoss):
    """Loss for standard addition: d1 + d2 == n"""

    def condition(self):
        return lambda d1, d2, n: torch.eq(d1.value + d2.value, n.value)


class mnist_sump_ltn_loss(BaseMNISTLTNLoss):
    """Loss for parity of sum: (d1 + d2) % 2 == n"""

    def condition(self):
        return lambda d1, d2, n: torch.eq((d1.value + d2.value) % 2, n.value)

def _generic_dsl_weights(n_images, concept_dim, output_dim, device):
    shape = [concept_dim] * n_images + [output_dim]
    return torch.nn.Parameter(torch.randn(shape).to(device))

def _generic_dsl_words_weights(n_images, concept_dim, output_dim, device):
    shape = [2**concept_dim] + [output_dim]
    return torch.nn.Parameter(torch.randn(shape).to(device))

def mnist_add_dsl_weights(n_images, concept_dim, output_dim, device):
    return _generic_dsl_weights(n_images, concept_dim, output_dim, device)

def mnist_sump_dsl_weights(n_images, concept_dim, output_dim, device):
    return _generic_dsl_weights(n_images, concept_dim, output_dim, device)


#
# =============== BOIA ===============
#


def boia_circuit():
    FS_w_q = _build_world_queries_matrix_FS()
    L_w_q = _build_world_queries_matrix_L()
    R_w_q = _build_world_queries_matrix_R()
    or_four_bits = _or_four_bits_circuit()

    # return all the circuits
    return FS_w_q, L_w_q, R_w_q, or_four_bits


def _build_world_queries_matrix_FS():

    possible_worlds = list(product(range(2), repeat=6))
    n_worlds = len(possible_worlds)
    n_queries = 4
    look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}

    w_q = torch.zeros(n_worlds, n_queries)  # (100, 20)
    for w in range(n_worlds):
        tl_green, follow, clear, tl_red, t_sign, obs = look_up[w]

        if tl_green + follow + clear > 0:
            if tl_green + tl_red == 2 or clear + obs == 2:
                pass
            elif tl_red + t_sign + obs > 0:
                w_q[w, 0] = 1  # not move
                w_q[w, 3] = 1  # stop
            else:
                w_q[w, 1] = 1  # move forward
                w_q[w, 2] = 1  # no-stop
        else:
            w_q[w, 0] = 1  # not move
            if tl_red + t_sign + obs > 0:
                w_q[w, 3] = 1  # stop
            else:
                w_q[w, 2] = 1  # no-stop
    return w_q


def _build_world_queries_matrix_L():
    possible_worlds = list(product(range(2), repeat=6))
    n_worlds = len(possible_worlds)
    n_queries = 2
    look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}

    w_q = torch.zeros(n_worlds, n_queries)
    for w in range(n_worlds):
        left_lane, tl_green, follow, no_left_lane, obs, left_solid_line = look_up[w]

        if left_lane + tl_green + follow + no_left_lane + obs + left_solid_line == 0:
            w_q[w, 0] = 0.5
            w_q[w, 1] = 0.5
        elif left_lane + tl_green + follow > 0:
            w_q[w, 1] = 1
        else:
            w_q[w, 0] = 1
    return w_q


def _build_world_queries_matrix_R():

    possible_worlds = list(product(range(2), repeat=6))
    n_worlds = len(possible_worlds)
    n_queries = 2
    look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}

    w_q = torch.zeros(n_worlds, n_queries)
    for w in range(n_worlds):
        right_lane, tl_green, follow, no_right_lane, obs, right_solid_line = look_up[w]

        if right_lane + tl_green + follow + no_right_lane + obs + right_solid_line == 0:
            w_q[w, 0] = 0.5
            w_q[w, 1] = 0.5
        elif right_lane + tl_green + follow > 0:
            if obs + right_solid_line + no_right_lane > 0:
                w_q[w, 0] = 1  # not move
            else:
                w_q[w, 1] = 1  # move
        else:
            w_q[w, 0] = 1  # not move
    return w_q


def _or_four_bits_circuit():
    four_bits_or = torch.cat((torch.zeros((16, 1)), torch.ones((16, 1))), dim=1).to(
        dtype=torch.float
    )
    four_bits_or[0] = torch.tensor([1, 0])

    return four_bits_or


class boia_ltn_loss(torch.nn.Module):
    """
    Base class for LTN-based MNIST logic losses.
    Encapsulates the common logic for variable creation and satisfaction aggregation.
    """

    def __init__(self, and_op, or_op, not_op, imp_op, exists_op, forall_op, equiv_op, sat_agg_op) -> None:
        super().__init__()
        self.and_op = and_op
        self.or_op = or_op
        self.not_op = not_op
        self.imp_op = imp_op
        self.exists_op = exists_op
        self.forall_op = forall_op
        self.equiv_op = equiv_op
        self.sat_agg_op = sat_agg_op

    def forward(self, conc_preds, actions):
        # Cut probabilities
        conc_preds = conc_preds[:, 0, :, 1]

        # Variables
        green_light = ltn.Variable("green_light", conc_preds[:, 0], add_batch_dim=False)
        follow = ltn.Variable("follow", conc_preds[:, 1], add_batch_dim=False)
        road_clear = ltn.Variable("road_clear", conc_preds[:, 2], add_batch_dim=False)
        red_light = ltn.Variable("red_light", conc_preds[:, 3], add_batch_dim=False)
        stop_sign = ltn.Variable("stop_sign", conc_preds[:, 4], add_batch_dim=False)
        car = ltn.Variable("car", conc_preds[:, 5], add_batch_dim=False)
        person = ltn.Variable("person", conc_preds[:, 6], add_batch_dim=False)
        rider = ltn.Variable("rider", conc_preds[:, 7], add_batch_dim=False)
        other_obstacle = ltn.Variable(
            "other_obstacle", conc_preds[:, 8], add_batch_dim=False
        )
        left_lane = ltn.Variable("left_lane", conc_preds[:, 9], add_batch_dim=False)
        left_green_light = ltn.Variable(
            "left_green_light", conc_preds[:, 10], add_batch_dim=False
        )
        left_follow = ltn.Variable("left_follow", conc_preds[:, 11], add_batch_dim=False)
        no_left_lane = ltn.Variable("no_left_lane", conc_preds[:, 12], add_batch_dim=False)
        left_obstacle = ltn.Variable(
            "left_obstacle", conc_preds[:, 13], add_batch_dim=False
        )
        left_solid_line = ltn.Variable(
            "left_solid_line", conc_preds[:, 14], add_batch_dim=False
        )
        right_lane = ltn.Variable("right_lane", conc_preds[:, 15], add_batch_dim=False)
        right_green_light = ltn.Variable(
            "right_green_light", conc_preds[:, 16], add_batch_dim=False
        )
        right_follow = ltn.Variable("right_follow", conc_preds[:, 17], add_batch_dim=False)
        no_right_lane = ltn.Variable(
            "no_right_lane", conc_preds[:, 18], add_batch_dim=False
        )
        right_obstacle = ltn.Variable(
            "right_obstacle", conc_preds[:, 19], add_batch_dim=False
        )
        right_solid_line = ltn.Variable(
            "right_solid_line", conc_preds[:, 20], add_batch_dim=False
        )
        move_forward = ltn.Variable("move_forward", actions[:, 0], add_batch_dim=False)
        stop = ltn.Variable("stop", actions[:, 1], add_batch_dim=False)
        turn_left = ltn.Variable("turn_left", actions[:, 2], add_batch_dim=False)
        turn_right = ltn.Variable("turn_right", actions[:, 3], add_batch_dim=False)

        # REDLIGHT: red_light ⇒ ¬green_light
        
        # phi1 = self.Forall(ltn.diag(red_light, green_light), self.Implies(red_light, self.Not(green_light)))
        phi1 = self.forall_op(
            ltn.diag(red_light, green_light), self.not_op(self.and_op(red_light, green_light))
        )
        
        # OBSTACLE: obstacle = car ∨ person ∨ rider ∨ other_obstacle
        def obstacle(c, p, r, o):
            return self.or_op(c, self.or_op(p, self.or_op(r, o)))
        
        # ROAD_CLEAR: road_clear ⇐⇒ ¬obstacle
        phi2 = self.forall_op(
            ltn.diag(road_clear, car, person, rider, other_obstacle),
            self.equiv_op(road_clear, self.not_op(obstacle(car, person, rider, other_obstacle))),
        )
        
        # MOVE_FORWARD: green_light ∨ follow ∨ clear ⇒ move_forward
        phi3 = self.forall_op(
            ltn.diag(green_light, follow, road_clear, move_forward),
            self.imp_op(self.or_op(green_light, self.or_op(follow, road_clear)), move_forward),
        )

        # STOP: red_light ∨ stop_sign ∨ obstacle ⇒ stop
        phi4 = self.forall_op(
            ltn.diag(red_light, stop_sign, car, person, rider, other_obstacle, stop),
            self.imp_op(
                self.or_op(
                    red_light,
                    self.or_op(stop_sign, obstacle(car, person, rider, other_obstacle)),
                ),
                stop,
            ),
        )
    
        phi5 = self.forall_op(
            ltn.diag(
                red_light,
                stop_sign,
                car,
                person,
                rider,
                other_obstacle,
                green_light,
                follow,
                road_clear,
            ),
            self.not_op(
                self.and_op(
                    self.or_op(
                        red_light,
                        self.or_op(stop_sign, obstacle(car, person, rider, other_obstacle)),
                    ),
                    self.or_op(green_light, self.or_op(follow, road_clear)),
                )
            ),
        )

        # LEFT CAN TURN: can_turn = left_lane ∨ left_green_lane ∨ left_follow
        def can_turn(lane, green_light, follow):
            return self.or_op(lane, self.or_op(green_light, follow))

        # LEFT CANNOT TURN: cannot_turn = no_left_lane ∨ left_obstacle ∨ left_solid_line
        def cannot_turn(no_lane, obstacle, solid_line):
            return self.or_op(no_lane, self.or_op(obstacle, solid_line))

        # TURN LEFT: can_turn ∧ ¬cannot_turn ⇒ turn_left
        phi6 = self.forall_op(
            ltn.diag(left_lane, left_green_light, left_follow, turn_left),
            self.equiv_op(can_turn(left_lane, left_green_light, left_follow), turn_left),
        )
        phi7 = self.forall_op(
            ltn.diag(no_left_lane, left_obstacle, left_solid_line, turn_left),
            self.equiv_op(
                self.not_op(cannot_turn(no_left_lane, left_obstacle, left_solid_line)),
                turn_left,
            ),
        )
        phi8 = self.forall_op(
            ltn.diag(right_lane, right_green_light, right_follow, turn_right),
            self.equiv_op(can_turn(right_lane, right_green_light, right_follow), turn_right),
        )
        phi9 = self.forall_op(
            ltn.diag(no_right_lane, right_obstacle, right_solid_line, turn_right),
            self.equiv_op(
                self.not_op(cannot_turn(no_right_lane, right_obstacle, right_solid_line)),
                turn_right,
            ),
        )

        log("phi1: " + str(phi1), "DEBUG")
        log("phi2: " + str(phi2), "DEBUG")
        log("phi3: " + str(phi3), "DEBUG")
        log("phi4: " + str(phi4), "DEBUG")
        log("phi5: " + str(phi5), "DEBUG")
        log("phi6: " + str(phi6), "DEBUG")
        log("phi7: " + str(phi7), "DEBUG")
        log("phi8: " + str(phi8), "DEBUG")
        log("phi9: " + str(phi9), "DEBUG")

        log(f"LTN loss: {1.0 - self.sat_agg_op(phi1, phi2, phi3, phi4, phi5, phi6, phi7, phi8, phi9)}", "DEBUG")

        return 1.0 - self.sat_agg_op(phi1, phi2, phi3, phi4, phi5, phi6, phi7, phi8, phi9)


def boia_dsl_weights(n_images, concept_dim, output_dim, device):
        return [
            # forward
            _generic_dsl_words_weights(
                n_images=n_images, 
                concept_dim=9,
                output_dim=2, 
                device=device
            ),
            # stop
            _generic_dsl_words_weights(
                n_images=n_images, 
                concept_dim=9,
                output_dim=2, 
                device=device
            ),
            # left-stop
            _generic_dsl_words_weights(
                n_images=n_images, 
                concept_dim=6,
                output_dim=2, 
                device=device
            ),
            # right-stop
            _generic_dsl_words_weights(
                n_images=n_images, 
                concept_dim=6,
                output_dim=2, 
                device=device
            )
        ]


##
# CHX
##

def chx_circuit():
    possible_worlds = list(product(range(2), repeat=4))
    n_worlds = len(possible_worlds)
    n_queries = 2
    look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}

    w_q = torch.zeros(n_worlds, n_queries)
    for w in range(n_worlds):
        fracture, pneumothorax, airspace_opacity, nodule_mass = look_up[w]

        # Healty vs not healty
        if fracture + pneumothorax + airspace_opacity + nodule_mass == 0:
            w_q[w, 1] = 1
        else:
            w_q[w, 0] = 1
    return w_q


class chx_ltn_loss(torch.nn.Module):
    def __init__(self, equiv_op, forall_op, not_op, exists_op, sat_agg_op) -> None:
        """
        Logic: The sample is 'Healthy' (label 0) if all 4 concepts are absent.
        Otherwise, it is 'Unhealthy' (label 1).
        """
        super().__init__()
        self.equiv_op = equiv_op
        self.forall_op = forall_op
        self.not_op = not_op
        self.sat_agg_op = sat_agg_op
        self.exists_op = exists_op

    def forward(self, pred_concepts, labels):
        x = ltn.Variable("x", pred_concepts[:, 0, :, 1] )
        l = ltn.Variable("l", labels)
        indices = ltn.Variable("indices", torch.arange(4))

        # Predicates
        is_present = ltn.Predicate(func=lambda c, idx: torch.gather(c, 1, idx.long()))
        is_healthy = ltn.Predicate(func=lambda l: (l == 0).float())

        # If healty <-> for all indices not present
        healty = self.forall_op(
            ltn.diag(x, l),
            self.equiv_op(
                is_healthy(l),
                self.forall_op(
                    [indices], 
                    self.not_op(is_present(x, indices))
                )
            )
        )

        # If malignant <-> exists at least one 
        malignant = self.forall_op(
            ltn.diag(x, l),
            self.equiv_op(
                self.not_op(is_healthy(l)),
                self.exists_op(
                    [indices], 
                    is_present(x, indices)
                )
            )
        )

        sat_agg = self.sat_agg_op(healty, malignant)
        log(f"LTN loss: {1 - sat_agg}", "DEBUG")
        return 1 - sat_agg

def chx_dsl_weights(n_images, concept_dim, output_dim, device):
    return _generic_dsl_words_weights(n_images, concept_dim, output_dim, device)
    
##
# DERMA
##


def derma_circuit():
    possible_worlds = list(product(range(2), repeat=7))
    n_worlds = len(possible_worlds)
    n_queries = 2
    look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}

    w_q = torch.zeros(n_worlds, n_queries)
    for w in range(n_worlds):
        actinic, basal, benign, derma, melano, melanoma, vascular = look_up[w]

        # Healty vs not healty
        if actinic + basal + melanoma >= 1:
            w_q[w, 1] = 1
        else:
            w_q[w, 0] = 1
    return w_q



class derma_ltn_loss(torch.nn.Module):
    def __init__(self, equiv_op, forall_op, not_op, exists_op, sat_agg_op) -> None:
        """
        Logic: 
        Malignant (1) <=> Concept 0 OR Concept 1 OR Concept 5 is True.
        Benign (0)    <=> NOT (Concept 0 OR Concept 1 OR Concept 5).
        """
        super().__init__()
        
        self.equiv_op = equiv_op
        self.forall_op = forall_op
        self.not_op = not_op
        self.sat_agg_op = sat_agg_op
        self.exists_op = exists_op

    def forward(self, pred_concepts, labels):
        x = ltn.Variable("x", pred_concepts[:, 0, :, 1] )
        l = ltn.Variable("l", labels)
        indices = ltn.Variable("indices", torch.tensor([0, 1, 5]))

        # Predicates
        is_present = ltn.Predicate(func=lambda c, idx: torch.gather(c, 1, idx.long()))
        is_healthy = ltn.Predicate(func=lambda l: (l == 1).float()) # NOTE: different from CHX

        # If healty <-> for all indices not present
        healty = self.forall_op(
            ltn.diag(x, l),
            self.equiv_op(
                is_healthy(l),
                self.forall_op(
                    [indices], 
                    self.not_op(is_present(x, indices))
                )
            )
        )

        # If malignant <-> exists at least one 
        malignant = self.forall_op(
            ltn.diag(x, l),
            self.equiv_op(
                self.not_op(is_healthy(l)),
                self.exists_op(
                    [indices], 
                    is_present(x, indices)
                )
            )
        )

        sat_agg = self.sat_agg_op(healty, malignant)
        log(f"LTN loss: {1 - sat_agg}", "DEBUG")
        return 1 - sat_agg


def derma_dsl_weights(n_images, concept_dim, output_dim, device):
    return _generic_dsl_words_weights(n_images, concept_dim, output_dim, device)


##
# CLEVR
##


def same_objects_circuit(n_images, n_colors=3, n_shapes=3, n_materials=2, n_sizes=2):
    dims = [n_colors, n_shapes, n_materials, n_sizes] * n_images

    # Generate all possible worlds (36 * 36 = 1296)
    possible_worlds = list(product(*[range(d) for d in dims]))

    n_worlds = len(possible_worlds)
    n_queries = 2  # 0: Different, 1: Same

    w_q = torch.zeros(n_worlds, n_queries)

    look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}

    for w in range(n_worlds):
        (
            color_1,
            shape_1,
            material_1,
            size_1,
            color_2,
            shape_2,
            material_2,
            size_2,
        ) = look_up[w]

        same = True

        if color_1 != color_2:
            same = False

        if shape_1 != shape_2:
            same = False

        if material_1 != material_2:
            same = False

        if size_1 != size_2:
            same = False

        if same:
            w_q[w, 1] = 1  # Match
        else:
            w_q[w, 0] = 1  # No Match

    return w_q
