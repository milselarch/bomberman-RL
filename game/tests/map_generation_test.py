import unittest
import game


class MyTestCase(unittest.TestCase):
    def test_map_generation_corners(self):
        grid = [row[:] for row in game.GRID_BASE]
        game.generate_map(grid)
        self.assertEqual(0, grid[1][1])
        self.assertEqual(0, grid[1][2])
        self.assertEqual(0, grid[2][1])

        length = len(grid)

        self.assertEqual(0, grid[length - 2][1])
        self.assertEqual(0, grid[length - 2][2])
        self.assertEqual(0, grid[length - 3][1])

        self.assertEqual(0, grid[1][length - 2])
        self.assertEqual(0, grid[1][length - 3])
        self.assertEqual(0, grid[2][length - 2])

        self.assertEqual(0, grid[length - 2][length - 2])
        self.assertEqual(0, grid[length - 2][length - 3])
        self.assertEqual(0, grid[length - 3][length - 2])


if __name__ == '__main__':
    unittest.main()

