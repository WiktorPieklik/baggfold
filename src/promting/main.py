from colorama import Fore

reset_colors = Fore.RESET


def print_themed(theme, msg: str) -> None:
    print(f"{theme}{msg}{reset_colors}")


def WARNING(msg: str) -> None:
    print_themed(Fore.YELLOW, msg)


def INFO(msg: str) -> None:
    print_themed(Fore.CYAN, msg)


def SUCCESS(msg: str) -> None:
    print_themed(Fore.GREEN, msg)


def IMPORTANT(msg: str) -> None:
    print_themed(Fore.RED, msg)
