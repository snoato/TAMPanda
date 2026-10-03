;; Access-19 v2 (tabletop-access-v2), rgnet/GNN variant with v1's derived blocking hint.
;; Identical to domain.pddl plus :derived-predicates and
;;   (blocking ?obj ?goal)  <=>  ?obj occupies a cell c with (same-column-front c ?goal).
;; Problem files are the same for both variants; (blocking ...) never appears in :init.

(define (domain tabletop-access-v2)
  (:requirements :strips :typing :negative-preconditions
                 :disjunctive-preconditions :existential-preconditions
                 :derived-predicates)

  (:types
    cell movable direction
  )

  (:constants
    north east - direction
  )

  (:predicates
    (adjacent ?dir - direction ?cel1 - cell ?cel2 - cell)
    (occupied ?cel - cell ?obj - movable)
    (empty ?cel - cell)
    (holding ?obj - movable)
    (gripper-empty)
    (same-column-front ?cel-front - cell ?cel-back - cell)
    (blocking ?obj - movable ?goal - cell)
  )

  (:derived (blocking ?obj - movable ?goal - cell)
    (exists (?c - cell)
      (and (occupied ?c ?obj)
           (same-column-front ?c ?goal))))

  (:action pick
    :parameters (?obj - movable ?cel - cell)
    :precondition (and
      (gripper-empty)
      (occupied ?cel ?obj)
    )
    :effect (and
      (not (gripper-empty))
      (not (occupied ?cel ?obj))
      (empty ?cel)
      (holding ?obj)
    )
  )

  (:action put
    :parameters (?obj - movable ?cel - cell)
    :precondition (and
      (holding ?obj)
      (empty ?cel)
    )
    :effect (and
      (gripper-empty)
      (not (holding ?obj))
      (not (empty ?cel))
      (occupied ?cel ?obj)
    )
  )
)
