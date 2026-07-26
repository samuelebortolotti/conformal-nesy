from itertools import product
from functools import reduce
import torch
import ltn
from conformal.general_utils import log

# ===============
# MNIST
# ===============


def mnist_circuit(sequence_len=2, n_digits=10, output_dim=19):
    possible_worlds = list(product(range(n_digits), repeat=sequence_len))
    n_worlds = len(possible_worlds)
    n_queries = len(range(0, output_dim))
    look_up = {i: c for i, c in enumerate(possible_worlds)}
    w_q = torch.zeros(n_worlds, n_queries)
    for w in range(n_worlds):
        digits = look_up[w]
        total = sum(digits)
        w_q[w, total] = 1
    return w_q


def mnist_sump_circuit(sequence_len=2, n_digits=10, output_dim=2):
    possible_worlds = list(product(range(n_digits), repeat=sequence_len))
    n_worlds = len(possible_worlds)
    n_queries = len(range(0, output_dim))
    look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}
    w_q = torch.zeros(n_worlds, n_queries)
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

    def __init__(self, and_op, exists_op, forall_op, n_outputs, n_images=2) -> None:
        super().__init__()
        self.and_op = and_op
        self.exists_op = exists_op
        self.forall_op = forall_op
        self.n_outputs = n_outputs
        self.n_images = n_images

    def condition(self):
        """Override this method to define the specific logic rule."""
        raise NotImplementedError

    def forward(self, pred_concepts, labels):
        # Variables representing the input images and the target label
        image_vars = [
            ltn.Variable(f"x{i}", pred_concepts[:, i]) for i in range(self.n_images)
        ]
        n = ltn.Variable("n", labels)

        # LTN predicate for digit classification
        digit_pred = ltn.Predicate(
            func=lambda digits, d_idx: torch.gather(digits, 1, d_idx)
        )

        # Digit variables for all possible digit values
        digit_vars = [
            ltn.Variable(f"d{i}", torch.arange(pred_concepts.shape[-1]))
            for i in range(self.n_images)
        ]

        # The core logical formula: Forall x,y,n: Exists d1,d2 such that (d1+d2 satisfy condition)
        sat_agg = self.forall_op(
            ltn.diag(*image_vars, n),
            self.exists_op(
                digit_vars,
                reduce(
                    lambda a, b: self.and_op(a, b),
                    [digit_pred(xi, di) for xi, di in zip(image_vars, digit_vars)],
                ),
                cond_vars=[*digit_vars, n],
                cond_fn=self.condition(),
            ),
        )

        log(f"LTN loss: {1 - sat_agg.value}", "DEBUG")

        return 1 - sat_agg.value


class mnist_add_ltn_loss(BaseMNISTLTNLoss):
    """Loss for standard addition: d1 + d2 == n"""

    def condition(self):
        return lambda *vars: torch.eq(
            sum(v.value for v in vars[:-1]),
            vars[-1].value,
        )


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

    def __init__(
        self, and_op, or_op, not_op, imp_op, exists_op, forall_op, equiv_op, sat_agg_op
    ) -> None:
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
        # Cut probabilities — shape (B, 21)
        conc_preds = conc_preds[:, 0, :, 1]

        # Concept variables — indices follow CONCEPTS_ORDER in boia.py:
        #  0:green_light  1:follow  2:clear  3:red_light  4:stop_sign
        #  5:car  6:person  7:rider  8:other_obstacle
        #  9:no_left_lane  10:left_obstacle  11:left_solid_line
        #  12:right_lane  13:right_green_light  14:right_follow
        #  15:no_right_lane  16:right_obstacle  17:right_solid_line
        #  18:left_lane  19:left_green_light  20:left_follow
        green_light    = ltn.Variable("green_light",    conc_preds[:, 0],  add_batch_dim=False)
        follow         = ltn.Variable("follow",         conc_preds[:, 1],  add_batch_dim=False)
        road_clear     = ltn.Variable("road_clear",     conc_preds[:, 2],  add_batch_dim=False)
        red_light      = ltn.Variable("red_light",      conc_preds[:, 3],  add_batch_dim=False)
        stop_sign      = ltn.Variable("stop_sign",      conc_preds[:, 4],  add_batch_dim=False)
        car            = ltn.Variable("car",            conc_preds[:, 5],  add_batch_dim=False)
        person         = ltn.Variable("person",         conc_preds[:, 6],  add_batch_dim=False)
        rider          = ltn.Variable("rider",          conc_preds[:, 7],  add_batch_dim=False)
        other_obstacle = ltn.Variable("other_obstacle", conc_preds[:, 8],  add_batch_dim=False)
        no_left_lane   = ltn.Variable("no_left_lane",   conc_preds[:, 9],  add_batch_dim=False)
        left_obstacle  = ltn.Variable("left_obstacle",  conc_preds[:, 10], add_batch_dim=False)
        left_solid_line= ltn.Variable("left_solid_line",conc_preds[:, 11], add_batch_dim=False)
        right_lane     = ltn.Variable("right_lane",     conc_preds[:, 12], add_batch_dim=False)
        right_green_light=ltn.Variable("right_green_light",conc_preds[:, 13],add_batch_dim=False)
        right_follow   = ltn.Variable("right_follow",   conc_preds[:, 14], add_batch_dim=False)
        no_right_lane  = ltn.Variable("no_right_lane",  conc_preds[:, 15], add_batch_dim=False)
        right_obstacle = ltn.Variable("right_obstacle", conc_preds[:, 16], add_batch_dim=False)
        right_solid_line=ltn.Variable("right_solid_line",conc_preds[:, 17],add_batch_dim=False)
        left_lane      = ltn.Variable("left_lane",      conc_preds[:, 18], add_batch_dim=False)
        left_green_light=ltn.Variable("left_green_light",conc_preds[:, 19],add_batch_dim=False)
        left_follow    = ltn.Variable("left_follow",    conc_preds[:, 20], add_batch_dim=False)

        # Label variables — label order: 0=STOP, 1=MOVE_FORWARD, 2=TURN_LEFT, 3=TURN_RIGHT
        stop         = ltn.Variable("stop",         actions[:, 0], add_batch_dim=False)
        move_forward = ltn.Variable("move_forward", actions[:, 1], add_batch_dim=False)
        turn_left    = ltn.Variable("turn_left",    actions[:, 2], add_batch_dim=False)
        turn_right   = ltn.Variable("turn_right",   actions[:, 3], add_batch_dim=False)

        # OBSTACLE aggregate: car ∨ person ∨ rider ∨ other_obstacle
        def obstacle(c, p, r, o):
            return self.or_op(c, self.or_op(p, self.or_op(r, o)))

        # phi1: red_light ⇒ ¬green_light
        phi1 = self.forall_op(
            ltn.diag(red_light, green_light),
            self.not_op(self.and_op(red_light, green_light)),
        )

        # phi2: road_clear ⇔ ¬obstacle
        phi2 = self.forall_op(
            ltn.diag(road_clear, car, person, rider, other_obstacle),
            self.equiv_op(
                road_clear, self.not_op(obstacle(car, person, rider, other_obstacle))
            ),
        )

        # phi3: (green_light ∧ follow ∧ road_clear) ⇒ move_forward
        # AND (not OR): all three conditions must hold before the axiom fires.
        # IMP (not equiv): the reverse direction would push move_forward toward the AND product
        # value (~0.125 when concepts are uncertain), conflicting with label BCE for forward
        # scenarios and causing F1 to decline after early epochs.
        # phi5 handles the stop-priority case without needing the reverse direction here.
        phi3 = self.forall_op(
            ltn.diag(green_light, follow, road_clear, move_forward),
            self.imp_op(
                self.and_op(green_light, self.and_op(follow, road_clear)), move_forward
            ),
        )

        # phi4: (red_light ∨ stop_sign ∨ obstacle) ⇒ stop
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

        # phi5: stop_cause ⇒ ¬move_forward (stop takes priority)
        # Note: we do NOT use the global mutual exclusion ¬(stop_cause ∧ move_cause) because
        # in BOIA green_light=1 AND obstacle=1 can legally coexist (stop takes priority).
        phi5 = self.forall_op(
            ltn.diag(
                red_light, stop_sign, car, person, rider, other_obstacle, move_forward
            ),
            self.imp_op(
                self.or_op(
                    red_light,
                    self.or_op(stop_sign, obstacle(car, person, rider, other_obstacle)),
                ),
                self.not_op(move_forward),
            ),
        )

        # Helper functions for turn logic
        def can_turn(lane, gl, fol):
            return self.or_op(lane, self.or_op(gl, fol))

        def cannot_turn(no_lane, obs, solid):
            return self.or_op(no_lane, self.or_op(obs, solid))

        # phi6: turn_left ⇔ (can_turn_left ∧ ¬cannot_turn_left)
        # Uses correct indices: enablers at 18/19/20, inhibitors at 9/10/11
        phi6 = self.forall_op(
            ltn.diag(
                left_lane, left_green_light, left_follow,
                no_left_lane, left_obstacle, left_solid_line,
                turn_left,
            ),
            self.equiv_op(
                self.and_op(
                    can_turn(left_lane, left_green_light, left_follow),
                    self.not_op(cannot_turn(no_left_lane, left_obstacle, left_solid_line)),
                ),
                turn_left,
            ),
        )

        # phi7: turn_right ⇔ (can_turn_right ∧ ¬cannot_turn_right)
        # Uses correct indices: enablers at 12/13/14, inhibitors at 15/16/17
        phi7 = self.forall_op(
            ltn.diag(
                right_lane, right_green_light, right_follow,
                no_right_lane, right_obstacle, right_solid_line,
                turn_right,
            ),
            self.equiv_op(
                self.and_op(
                    can_turn(right_lane, right_green_light, right_follow),
                    self.not_op(cannot_turn(no_right_lane, right_obstacle, right_solid_line)),
                ),
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

        log(
            f"LTN loss: {1.0 - self.sat_agg_op(phi1, phi2, phi3, phi4, phi5, phi6, phi7)}",
            "DEBUG",
        )

        return 1.0 - self.sat_agg_op(phi1, phi2, phi3, phi4, phi5, phi6, phi7)


def boia_dsl_weights(n_images, concept_dim, output_dim, device):
    return [
        # forward
        _generic_dsl_words_weights(
            n_images=n_images, concept_dim=9, output_dim=2, device=device
        ),
        # stop
        _generic_dsl_words_weights(
            n_images=n_images, concept_dim=9, output_dim=2, device=device
        ),
        # left-stop
        _generic_dsl_words_weights(
            n_images=n_images, concept_dim=6, output_dim=2, device=device
        ),
        # right-stop
        _generic_dsl_words_weights(
            n_images=n_images, concept_dim=6, output_dim=2, device=device
        ),
    ]


##
# CHX
##


def chx_circuit(multi_class=False):
    possible_worlds = list(product(range(2), repeat=4))
    n_worlds = len(possible_worlds)
    n_queries = 2 if not multi_class else 5
    look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}

    w_q = torch.zeros(n_worlds, n_queries)
    for w in range(n_worlds):
        fracture, pneumothorax, airspace_opacity, nodule_mass = look_up[w]

        if not multi_class:
            # Healty vs not healty
            if fracture + pneumothorax + airspace_opacity + nodule_mass == 0:
                w_q[w, 1] = 1
            else:
                w_q[w, 0] = 1
        else:
            active = sum([fracture, pneumothorax, airspace_opacity, nodule_mass])
            w_q[w, active] = 1
    return w_q


class chx_ltn_loss(torch.nn.Module):
    def __init__(
        self, equiv_op, forall_op, not_op, exists_op, sat_agg_op, and_op, multi_class
    ) -> None:
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
        self.multi_class = multi_class
        self.and_op = and_op

    def forward_healthy_malignant(self, pred_concepts, labels):
        x = ltn.Variable("x", pred_concepts[:, 0, :, 1])
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
                self.forall_op([indices], self.not_op(is_present(x, indices))),
            ),
        )

        # If malignant <-> exists at least one
        malignant = self.forall_op(
            ltn.diag(x, l),
            self.equiv_op(
                self.not_op(is_healthy(l)),
                self.exists_op([indices], is_present(x, indices)),
            ),
        )

        sat_agg = self.sat_agg_op(healty, malignant)
        log(f"LTN loss: {1 - sat_agg}", "DEBUG")
        return 1 - sat_agg

    def forward(self, pred_concepts, labels):
        if self.multi_class:
            return self.forward_multi_class(pred_concepts, labels)
        return self.forward_healthy_malignant(pred_concepts, labels)

    # def forward_multi_class(self, pred_concepts, labels):
    #     # Apprendibili stanno in predicate or in function.
    #     x = ltn.Variable("x", pred_concepts[:, 0, :, 1]) # Rete Neurale
    #     l = ltn.Variable("l", labels)

    #     # Symptom variables
    #     symptom_vars = [
    #         ltn.Variable(f"s{i}", torch.arange(2)) for i in range(4)  # binary: 0 or 1
    #     ]

    #     # predicates
    #     is_present = ltn.Predicate(
    #         func=lambda c, idx: 
    #         torch.gather(c, dim=1, index=idx.long())
    #     )  

    #     # label is the sum of symtom
    #     def condition():
    #         return lambda *vars: torch.eq(
    #             sum(v.value for v in vars[:-1]), vars[-1].value
    #         )

    #     # Sat Agg
    #     sat_agg = self.forall_op(
    #         ltn.diag(x, l),
    #         self.exists_op(
    #             symptom_vars,
    #             self.and_op(
    #                 self.and_op(
    #                     self.and_op(
    #                         is_present(x, symptom_vars[0]),
    #                         is_present(x, symptom_vars[1]),
    #                     ),
    #                     is_present(x, symptom_vars[2]),
    #                 ),
    #                 is_present(x, symptom_vars[3]),
    #             ),
    #             cond_vars=[*symptom_vars, l],
    #             cond_fn=condition(),
    #         ),
    #     )
    #     log(f"LTN loss: {1 - sat_agg.value}", "DEBUG")
    #     return 1 - sat_agg.value

    def forward_multi_class(self, pred_concepts, labels):
        x = ltn.Variable("x", pred_concepts[:, 0, :, 1])  # [B, 4]: P(concept_j=1)
        l = ltn.Variable("l", labels)

        # s_i ∈ {0, 1}: whether symptom i is absent or present
        # float() required because s_i is used in arithmetic below
        symptom_vars = [
            ltn.Variable(f"s{i}", torch.arange(2).float()) for i in range(4)
        ]

        # Per-concept predicate: P(concept_j = s_j)
        # c = x after LTN expansion → (..., 4) last dim; s = s_j after LTN expansion
        concept_preds = [
            ltn.Predicate(func=lambda c, s, j=j: s * c[..., j:j+1] + (1 - s) * (1 - c[..., j:j+1]))
            for j in range(4)
        ]

        def condition():
            return lambda *vars: torch.eq(
                sum(v.value for v in vars[:-1]), vars[-1].value
            )

        sat_agg = self.forall_op(
            ltn.diag(x, l),
            self.exists_op(
                symptom_vars,
                self.and_op(
                    self.and_op(
                        self.and_op(
                            concept_preds[0](x, symptom_vars[0]),
                            concept_preds[1](x, symptom_vars[1]),
                        ),
                        concept_preds[2](x, symptom_vars[2]),
                    ),
                    concept_preds[3](x, symptom_vars[3]),
                ),
                cond_vars=[*symptom_vars, l],
                cond_fn=condition(),
            ),
        )
        log(f"LTN loss: {1 - sat_agg.value}", "DEBUG")
        return 1 - sat_agg.value


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
        x = ltn.Variable("x", pred_concepts[:, 0, :, 1])
        l = ltn.Variable("l", labels)
        indices = ltn.Variable("indices", torch.tensor([0, 1, 5]))

        # Predicates
        is_present = ltn.Predicate(func=lambda c, idx: torch.gather(c, 1, idx.long()))
        is_healthy = ltn.Predicate(
            func=lambda l: (l == 1).float()
        )  # NOTE: different from CHX

        # If healty <-> for all indices not present
        healty = self.forall_op(
            ltn.diag(x, l),
            self.equiv_op(
                is_healthy(l),
                self.forall_op([indices], self.not_op(is_present(x, indices))),
            ),
        )

        # If malignant <-> exists at least one
        malignant = self.forall_op(
            ltn.diag(x, l),
            self.equiv_op(
                self.not_op(is_healthy(l)),
                self.exists_op([indices], is_present(x, indices)),
            ),
        )

        sat_agg = self.sat_agg_op(healty, malignant)
        log(f"LTN loss: {1 - sat_agg}", "DEBUG")
        return 1 - sat_agg


def derma_dsl_weights(n_images, concept_dim, output_dim, device):
    return _generic_dsl_words_weights(n_images, concept_dim, output_dim, device)


def rival_dsl_weights(n_images, concept_dim, output_dim, device):
    return _generic_dsl_words_weights(1, concept_dim, output_dim, device)


def cifar_dsl_weights(n_images, concept_dim, output_dim, device):
    return _generic_dsl_words_weights(1, concept_dim, output_dim, device)


def cebab_dsl_weights(n_images, concept_dim, output_dim, device):
    return _generic_dsl_weights(n_images, concept_dim, output_dim, device)


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


##
# CIFAR
##


def cifar_circuit():
    possible_worlds = list(product(range(2), repeat=7))
    n_worlds = len(possible_worlds)
    n_queries = 10
    look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}

    w_q = torch.zeros(n_worlds, n_queries)
    for w in range(n_worlds):
        wheels, metallic, wings, animal, hairy, horns, long_snout = look_up[w]

        is_vehicle = metallic * (1 - animal)
        is_living = animal * (1 - metallic)

        plane = is_vehicle * wings
        car_truck = is_vehicle * wheels * (1 - wings)
        ship = is_vehicle * (1 - wheels) * (1 - wings)
        bird = is_living * wings
        frog = is_living * (1 - hairy)
        deer = is_living * (1 - wings) * hairy * horns
        cat = is_living * (1 - wings) * hairy * (1 - horns) * (1 - long_snout)
        dog_equine = is_living * (1 - wings) * hairy * (1 - horns) * long_snout

        if plane:
            w_q[w, 0] = 1
        elif car_truck or wheels:
            w_q[w, 1] = 0.5
            w_q[w, 9] = 0.5
        elif bird:
            w_q[w, 2] = 1
        elif cat:
            w_q[w, 3] = 1
        elif deer:
            w_q[w, 4] = 1
        elif dog_equine:
            w_q[w, 5] = 0.5
            w_q[w, 7] = 0.5
        elif frog:
            w_q[w, 6] = 1
        elif ship:
            w_q[w, 8] = 1
        elif metallic:  # plane, car, truck and ship with equal prob
            w_q[w, 0] = 0.25
            w_q[w, 1] = 0.25
            w_q[w, 9] = 0.25
            w_q[w, 8] = 0.25
        elif wings:  # plane or bird
            w_q[w, 1] = 1
            w_q[w, 2] = 1
        elif hairy:
            w_q[w, 3] = 0.25
            w_q[w, 4] = 0.25
            w_q[w, 5] = 0.25
            w_q[w, 7] = 0.25
        elif long_snout:
            w_q[w, 4] = 1.0 / 3
            w_q[w, 5] = 1.0 / 3
            w_q[w, 7] = 1.0 / 3
        else:  # can be anything
            w_q[w, :] = 1.0 / 10
    return w_q


class cifar_ltn_loss(torch.nn.Module):
    def __init__(self, equiv_op, forall_op, not_op, and_op, sat_agg_op, or_op):
        super().__init__()
        self.equiv_op = equiv_op
        self.forall_op = forall_op
        self.not_op = not_op
        self.and_op = and_op
        self.sat_agg_op = sat_agg_op
        self.or_op = or_op
        self.concept_names = ["whl", "met", "wng", "ani", "hai", "hrn", "snt"]

    def forward(self, pred_concepts, labels):
        c = ltn.Variable("c", pred_concepts[:, 0, :, 1])
        l = ltn.Variable("l", labels)
        class_targets = {i: ltn.Constant(torch.tensor([i])) for i in range(10)}

        is_present = ltn.Predicate(func=lambda c, idx: torch.gather(c, 1, idx.long()))
        is_class = ltn.Predicate(
            func=lambda l_val, target: (l_val == target).float().unsqueeze(-1)
        )
        idx = {
            name: ltn.Constant(torch.tensor([i]))
            for i, name in enumerate(self.concept_names)
        }

        # helper: n-ary AND
        def And_n(*args):
            out = args[0]
            for x in args[1:]:
                out = self.and_op(out, x)
            return out

        # --------- Class labels (CIFAR-10 indices) ---------
        # 0: airplane, 1: automobile, 2: bird, 3: cat, 4: deer,
        # 5: dog, 6: frog, 7: horse, 8: ship, 9: truck

        # plane  <->  met ∧ ¬ani ∧ wng
        plane = self.forall_op(
            ltn.diag(c, l),
            self.equiv_op(
                is_class(l, class_targets[0]),
                And_n(
                    is_present(c, idx["met"]),
                    self.not_op(is_present(c, idx["ani"])),
                    is_present(c, idx["wng"]),
                ),
            ),
        )

        # automobile ∨ truck  <->  whl ∧ ¬wng ∧ met ∧ ¬ani
        # (indistinguishable pair)
        car_truck = self.forall_op(
            ltn.diag(c, l),
            self.equiv_op(
                self.or_op(
                    is_class(l, class_targets[1]),  # automobile
                    is_class(l, class_targets[9]),  # truck
                ),
                And_n(
                    is_present(c, idx["whl"]),
                    self.not_op(is_present(c, idx["wng"])),
                    is_present(c, idx["met"]),
                    self.not_op(is_present(c, idx["ani"])),
                ),
            ),
        )

        # bird  <->  ani ∧ ¬met ∧ wng
        bird = self.forall_op(
            ltn.diag(c, l),
            self.equiv_op(
                is_class(l, class_targets[2]),
                And_n(
                    is_present(c, idx["ani"]),
                    self.not_op(is_present(c, idx["met"])),
                    is_present(c, idx["wng"]),
                ),
            ),
        )

        # frog  <->  ani ∧ ¬met ∧ ¬hai
        frog = self.forall_op(
            ltn.diag(c, l),
            self.equiv_op(
                is_class(l, class_targets[6]),
                And_n(
                    is_present(c, idx["ani"]),
                    self.not_op(is_present(c, idx["met"])),
                    self.not_op(is_present(c, idx["hai"])),
                ),
            ),
        )

        # deer  <->  ani ∧ ¬met ∧ ¬wng ∧ hai ∧ hrn
        deer = self.forall_op(
            ltn.diag(c, l),
            self.equiv_op(
                is_class(l, class_targets[4]),
                And_n(
                    is_present(c, idx["ani"]),
                    self.not_op(is_present(c, idx["met"])),
                    self.not_op(is_present(c, idx["wng"])),
                    is_present(c, idx["hai"]),
                    is_present(c, idx["hrn"]),
                ),
            ),
        )

        # cat  <->  ani ∧ ¬met ∧ ¬wng ∧ hai ∧ ¬hrn ∧ ¬snt
        cat = self.forall_op(
            ltn.diag(c, l),
            self.equiv_op(
                is_class(l, class_targets[3]),
                And_n(
                    is_present(c, idx["ani"]),
                    self.not_op(is_present(c, idx["met"])),
                    self.not_op(is_present(c, idx["wng"])),
                    is_present(c, idx["hai"]),
                    self.not_op(is_present(c, idx["hrn"])),
                    self.not_op(is_present(c, idx["snt"])),
                ),
            ),
        )

        # dog ∨ horse  <->  ani ∧ ¬met ∧ ¬wng ∧ hai ∧ ¬hrn ∧ snt
        # (indistinguishable pair)
        dog_horse = self.forall_op(
            ltn.diag(c, l),
            self.equiv_op(
                self.or_op(
                    is_class(l, class_targets[5]),  # dog
                    is_class(l, class_targets[7]),  # horse
                ),
                And_n(
                    is_present(c, idx["ani"]),
                    self.not_op(is_present(c, idx["met"])),
                    self.not_op(is_present(c, idx["wng"])),
                    is_present(c, idx["hai"]),
                    self.not_op(is_present(c, idx["hrn"])),
                    is_present(c, idx["snt"]),
                ),
            ),
        )

        # ship  <->  met ∧ ¬ani ∧ ¬whl ∧ ¬wng
        ship = self.forall_op(
            ltn.diag(c, l),
            self.equiv_op(
                is_class(l, class_targets[8]),
                And_n(
                    is_present(c, idx["met"]),
                    self.not_op(is_present(c, idx["ani"])),
                    self.not_op(is_present(c, idx["whl"])),
                    self.not_op(is_present(c, idx["wng"])),
                ),
            ),
        )

        sat_agg = self.sat_agg_op(
            plane, car_truck, bird, cat, deer, dog_horse, frog, ship
        )

        return 1 - sat_agg


#
# CEBAB
##


def cebab_circuit():
    # 5 concepts, 4 values
    # concepts: "food", "service", "noise", "ambiance", "rating"
    # values: "positive", "negative", "neutral", "unknown"
    possible_worlds = list(product(range(4), repeat=5))
    n_worlds = len(possible_worlds)
    n_queries = 5  # "positive", "negative", "neutral", "unknown", "conflict"
    look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}
    w_q = torch.zeros(n_worlds, n_queries)

    for w in range(n_worlds):
        food, service, noice, ambiance, rating = look_up[w]
        n_pos = sum([e == 0 for e in [food, service, noice, ambiance, rating]])
        n_neg = sum([e == 1 for e in [food, service, noice, ambiance, rating]])
        n_neu = sum([e == 2 for e in [food, service, noice, ambiance, rating]])
        n_unk = sum([e == 3 for e in [food, service, noice, ambiance, rating]])

        # majority is both positive and negative
        if n_pos >= max(n_neu, n_unk) and n_pos == n_neg:
            w_q[w, 4] = 1
        # majority positive or (positive-unk, positive-neut)
        elif n_pos >= max(n_neg, n_neu, n_unk):
            w_q[w, 0] = 1
        # majority negative  or (negative-unk, negative-neut)
        elif n_neg >= max(n_pos, n_neu, n_unk):
            w_q[w, 1] = 1
        # majority neutral
        elif n_neu >= max(n_pos, n_neg, n_unk):
            w_q[w, 2] = 1
        # majority unknown
        else:
            w_q[w, 3] = 1
    return w_q


class cebab_ltn_loss(torch.nn.Module):
    def __init__(self, equiv_op, forall_op, not_op, and_op, sat_agg_op, or_op):
        super().__init__()

        self.equiv_op = equiv_op
        self.forall_op = forall_op
        self.not_op = not_op
        self.and_op = and_op
        self.sat_agg_op = sat_agg_op
        self.or_op = or_op

    def forward(self, pred_concepts, labels):
        # sums
        n_pos = pred_concepts[..., 0].sum(dim=1)
        n_neg = pred_concepts[..., 1].sum(dim=1)
        n_neu = pred_concepts[..., 2].sum(dim=1)
        n_unk = pred_concepts[..., 3].sum(dim=1)

        v_pos = ltn.Variable("pos", n_pos)
        v_neg = ltn.Variable("neg", n_neg)
        v_neu = ltn.Variable("neu", n_neu)
        v_unk = ltn.Variable("unk", n_unk)
        l = ltn.Variable("l", labels)

        class_targets = {i: ltn.Constant(torch.tensor([i])) for i in range(5)}

        is_class = ltn.Predicate(
            func=lambda l_val, target: (l_val == target).float().unsqueeze(-1)
        )

        ge = ltn.Predicate(func=lambda a, b: torch.sigmoid((a - b)))

        eq = ltn.Predicate(func=lambda a, b: torch.exp(-torch.abs(a - b)))

        # Conflict: pos >= neu AND pos >= unk AND pos == neg
        is_conflict = self.forall_op(
            ltn.diag(v_pos, v_neu, v_unk, v_neg, l),
            self.equiv_op(
                is_class(l, class_targets[4]),
                self.and_op(
                    ge(v_pos, v_neu), self.and_op(ge(v_pos, v_unk), eq(v_pos, v_neg))
                ),
            ),
        )

        # Positive: pos > neg AND pos >= neu AND pos >= unk
        is_positive = self.forall_op(
            ltn.diag(v_pos, v_neu, v_unk, v_neg, l),
            self.equiv_op(
                is_class(l, class_targets[0]),
                self.and_op(
                    self.and_op(ge(v_pos, v_neg), self.not_op(eq(v_pos, v_neg))),
                    self.and_op(ge(v_pos, v_neu), ge(v_pos, v_unk)),
                ),
            ),
        )

        # Negative: neg > pos AND neg >= neu AND neg >= unk
        is_negative = self.forall_op(
            ltn.diag(v_pos, v_neu, v_unk, v_neg, l),
            self.equiv_op(
                is_class(l, class_targets[1]),
                self.and_op(
                    self.and_op(ge(v_neg, v_pos), self.not_op(eq(v_neg, v_pos))),
                    self.and_op(ge(v_neg, v_neu), ge(v_neg, v_unk)),
                ),
            ),
        )

        # Neutral: neu > pos AND neu > neg AND neu >= unk
        is_neutral = self.forall_op(
            ltn.diag(v_pos, v_neu, v_unk, v_neg, l),
            self.equiv_op(
                is_class(l, class_targets[2]),
                self.and_op(
                    self.and_op(ge(v_neu, v_pos), self.not_op(eq(v_neu, v_pos))),
                    self.and_op(
                        self.and_op(ge(v_neu, v_neg), self.not_op(eq(v_neu, v_neg))),
                        ge(v_neu, v_unk),
                    ),
                ),
            ),
        )

        # Unknown: unk > pos AND unk > neg AND unk > neu
        is_unknown = self.forall_op(
            ltn.diag(v_pos, v_neu, v_unk, v_neg, l),
            self.equiv_op(
                is_class(l, class_targets[3]),
                self.and_op(
                    self.and_op(ge(v_unk, v_pos), self.not_op(eq(v_unk, v_pos))),
                    self.and_op(
                        self.and_op(ge(v_unk, v_neg), self.not_op(eq(v_unk, v_neg))),
                        self.and_op(ge(v_unk, v_neu), self.not_op(eq(v_unk, v_neu))),
                    ),
                ),
            ),
        )

        sat_agg = self.sat_agg_op(
            is_conflict, is_positive, is_negative, is_neutral, is_unknown
        )

        return 1.0 - sat_agg
