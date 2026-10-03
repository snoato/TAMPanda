;; Access-19 v2 (tabletop-access-v2): v1's two-action domain on the v2 shelf (5 x 7 cells per level).
;; Filter mode: physical feasibility (reachability, blocking by other objects) is NOT modelled here;
;; the v2 feasibility checker (tampanda access19_v2.checker, a grasp-library lookup) prunes actions.

(define (domain tabletop-access-v2)
  (:requirements :strips :typing :negative-preconditions)

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
  )

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
