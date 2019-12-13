import numpy as np
from numba import njit


class Santorini:
    """
    A fast reimplement version of Santorini. 

    Args:
        auto_invert: all players will always see -1, -2 as their workers, useful for self playing
        superpower: whether to enable superpowers
        n_win_dome: who puts the n-th dome wins (with superpower enabled)
    """
    def __init__(
            self,
            board_dim=(5, 5),
            starting_parts=np.array([0, 22, 18, 14, 18]),
            winning_floor=3,
            auto_invert: bool = True,
            superpower: bool = False,
            n_win_dome: int = 5,
    ):
        self.board_dim = board_dim
        self.starting_parts = starting_parts
        self.winning_floor = winning_floor
        self.auto_invert = auto_invert
        self.superpower = superpower
        self.n_win_dome = n_win_dome

        self.moves = ['q', 'w', 'e', 'a', 'd', 'z', 'x', 'c']
        self.builds = ['q', 'w', 'e', 'a', 'd', 'z', 'x', 'c']

        # key to coordinates
        self.ktoc = {
            'q': np.array((-1, -1)),
            'w': np.array((-1, 0)),
            'e': np.array((-1, 1)),
            'a': np.array((0, -1)),
            'd': np.array((0, 1)),
            'z': np.array((1, -1)),
            'x': np.array((1, 0)),
            'c': np.array((1, 1))
        }
        # possbile moves, we base them on the first player
        self.itoa = [(w, m, b) for w in [-1, -2] for m in self.moves
                     for b in self.builds]
        # action to index
        self.atoi = {action: index for index, action in enumerate(self.itoa)}
        # worker ot index
        self.wtoi = {-1: 0, -2: 1, 1: 2, 2: 3}
        
        self.action_size = len(self.itoa)

        # cache the legal move query
        self._legal_move_cache = None

        self.reset()

    def set_state(self, board, workers, parts, current_player, done):
        """set the env to a specific point in time, useful for planning"""
        # note: reset will also clear the cache
        self.reset()
        self._board = board.copy()
        self._workers = workers.copy()
        self._parts = parts.copy()
        self.current_player = current_player
        self._done = done

    def get_state(self):
        """this state is used only for save and load"""
        return {
            'board': self._board,
            'workers': self._workers,
            'parts': self._parts,
            'current_player': self.current_player,
            'done': self._done,
        }

    def reset(self):
        self.current_player = -1
        self._parts = self.starting_parts.copy()
        self._workers = np.array([
            (0, 2),
            (4, 2),
            (2, 0),
            (2, 4),
        ])
        self._board = np.zeros(self.board_dim, dtype=np.int64)
        self._done = False
        # invalidate the cache
        self._legal_move_cache = None
        return self._state()

    def _state(self):
        workers = {k: v for k, v in zip(self.wtoi.keys(), self._workers)}
        w_board = np.zeros_like(self._board)
        for k, v in workers.items():
            w_board[tuple(v)] = k

        if self.auto_invert:
            # auto invert the worker number
            # so that each player will see themselves playing -1, -2 workers
            # note: this does not change the underlying data
            w_board = -self.current_player * w_board

        p_board = np.zeros_like(self._board)
        for i, p in enumerate(self._parts):
            p_board[i, i] = p

        return np.stack([self._board, w_board, p_board])

    def legal_moves(self):
        """get possible moves
        Returns:
            list of move indexes (int)
        """
        assert not self._done, "must reset"
        if self._legal_move_cache is None:
            # all possbile moves
            out = []
            for i, (worker, mdir, bdir) in enumerate(self.itoa):
                # the worker sign is not important
                # we use the sign from the current player
                worker = self.current_player * abs(worker)
                wid = self.wtoi[worker]  # worker id
                mdir = self.ktoc[mdir]  # move direction
                bdir = self.ktoc[bdir]  # build direction
                correct, moved, built, part = _check(
                    wid,
                    mdir,
                    bdir,
                    workers=self._workers,
                    board=self._board,
                    parts=self._parts,
                    winning_floor=self.winning_floor)
                if correct:
                    out.append(i)
            self._legal_move_cache = out
        return self._legal_move_cache

    def step(self, action):
        """
        Returns:
            (next state, reward, done, _)
            next state: array [3, 5, 5]; 
                        0: board, 1: workers, 2: part counts
            reward: 1 or 0, 1 if the player (who takes the action) wins
            done: True or False
        """
        assert not self._done, "must reset"
        worker, mdir, bdir = self.itoa[action]

        # the worker sign is not important
        # we use the sign from the current player
        worker = self.current_player * abs(worker)
        wid = self.wtoi[worker]
        mdir = self.ktoc[mdir]
        bdir = self.ktoc[bdir]
        correct, moved, built, part = _check(wid,
                                             mdir,
                                             bdir,
                                             workers=self._workers,
                                             board=self._board,
                                             parts=self._parts,
                                             winning_floor=self.winning_floor)
        if correct:
            reward, done = 0., False
            # move
            self._workers[wid] = moved

            # if win (standing on the third floor)
            # note: on winning move, no building
            if self._board[moved[0], moved[1]] == self.winning_floor:
                reward = 1.
                done = True
            else:
                # build and decrease the part count
                self._board[built[0], built[1]] = part
                self._parts[part] -= 1
                # superpower
                # the player who puts the n-th dome wins
                if self.superpower:
                    n_dome = (self._board == 4).sum()
                    if n_dome == self.n_win_dome:
                        reward = 1.
                        done = True
        else:
            raise ValueError('illegal move')

        # switch the player
        self.current_player *= -1
        # invalidate the cache
        # only when the player changes, after step
        self._legal_move_cache = None

        # check if the next player is not possible to move
        # the previous wins
        # note: must switch the player first
        # the legal move can be "reused" (as a cache) for the next legal move query
        if len(self.legal_moves()) == 0:
            reward = 1.
            done = True

        self._done = done
        # state is the next state (that is for the next player)
        return self._state(), reward, done, {}
    
    def tostring(self, state):
        return state.tostring()


@njit
def _walkable(
        wid: int,
        dir: np.ndarray,
        workers: np.ndarray,
        board: np.ndarray,
):
    """check if the move is valid"""
    pos = None
    # check boundary
    src = workers[wid]
    new = src + dir
    board_dim = board.shape
    if not (0 <= new[0] < board_dim[0]): return False, pos
    if not (0 <= new[1] < board_dim[1]): return False, pos

    # not a dome
    tgt = board[new[0], new[1]]
    if tgt == 4: return False, pos

    # not too high
    cur = board[src[0], src[1]]
    if tgt > cur + 1: return False, pos

    # no other worker
    for i in range(len(workers)):
        if i != wid:
            oth = workers[i]
            if oth[0] == new[0] and oth[1] == new[1]:
                return False, pos

    # return
    pos = new
    return True, pos


@njit
def _buildable(
        src: np.ndarray,
        dir: np.ndarray,
        wid: int,
        workers: np.ndarray,
        board: np.ndarray,
        parts: np.ndarray,
):
    """check if the build is valid"""
    part = None
    pos = None

    # check boundary
    new = src + dir
    board_dim = board.shape
    if not (0 <= new[0] < board_dim[0]): return False, part, pos
    if not (0 <= new[1] < board_dim[1]): return False, part, pos

    # not a dome
    tgt = board[new[0], new[1]]
    if tgt == 4: return False, part, pos

    # no other worker
    for i in range(len(workers)):
        if i != wid:
            oth = workers[i]
            if oth[0] == new[0] and oth[1] == new[1]:
                return False, part, pos

    # check parts
    # must be able to build
    part = tgt + 1
    if parts[tgt + 1] == 0: return False, part, pos

    # return
    pos = new
    return True, part, pos


@njit
def _check(
        wid: int,
        mdir: np.ndarray,
        bdir: np.ndarray,
        workers: np.ndarray,
        board: np.ndarray,
        parts: np.ndarray,
        winning_floor: int,
):
    """check movable and buildable
    Returns:
        (is_valid, moved_position, built_position, built_part)
    """
    walkable, moved = _walkable(wid, mdir, workers=workers, board=board)
    if walkable:
        # moved and won, no check for building
        if board[moved[0], moved[1]] == winning_floor:
            return True, moved, None, None

        buildable, part, built = _buildable(moved,
                                            bdir,
                                            wid=wid,
                                            workers=workers,
                                            board=board,
                                            parts=parts)
        if buildable:
            return True, moved, built, part
    return False, None, None, None
