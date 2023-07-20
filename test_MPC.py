"""
A test of the MPC algorithm.

This test is performed at this level as it involves too many modules and is
close to a finished product.
"""

import sys
sys.path.append("src")

# Modules from this repository
import test_systems

# Modules from the standard library


# Other modules

if __name__ == "__main__":


    nordic = test_systems.get_dynamic_Nordic()

    print(
        nordic.generate_table(
            show_lines=False,
            show_transformers=False,
        )
    )

    nordic = test_systems.get_Nordic_with_DERAs(penetration=0.5)

    print(nordic.generate_table(show_lines=False, show_transformers=False))