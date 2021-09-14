import pygame
import sys
import scripts.tools as tools
import scripts.settings as settings
import scripts.game as game


# base class for menu
class Menu:
    def __init__(self, main):
        self.main = main

        self.current_menu = 'MainMenu'

        self.main_menu = MainMenu(self)
        self.shop = Shop(self)
        self.credits = Credits(self)

        # background image
        self.background_image = pygame.image.load('assets/sprites/BackdropMain.png')
        self.background_image = pygame.transform.scale2x(self.background_image)

        # music
        pygame.mixer.music.load('assets/sounds/MainMenu.wav')
        pygame.mixer.music.set_volume(self.main.global_volume * self.main.music_volume)
        pygame.mixer.music.play(-1, fade_ms=2300)

        # buttons images
        self.sound_on_image = pygame.image.load('assets/sprites/ButtonSoundOn.png').convert_alpha()
        self.sound_off_image = pygame.image.load('assets/sprites/ButtonSoundOff.png').convert_alpha()
        self.music_on_image = pygame.image.load('assets/sprites/ButtonMusicOn.png').convert_alpha()
        self.music_off_image = pygame.image.load('assets/sprites/ButtonMusicOff.png').convert_alpha()

        # buttons
        self.button_sound = tools.Button(self.main.screen, self.sound_on_image, (1206, 20))
        self.button_music = tools.Button(self.main.screen, self.music_on_image, (1144, 20))


    def update_menu(self, main):
        self.main = main

        if self.current_menu == 'MainMenu':
            self.main_menu.update()
        elif self.current_menu == 'Shop':
            self.shop.update()
        elif self.current_menu == 'Credits':
            self.credits.update()

        pygame.mixer.music.set_volume(self.main.global_volume * self.main.music_volume)

        # if music is not playing
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.load('assets/sounds/MainMenu.wav')
            pygame.mixer.music.play(-1, fade_ms=2300)

        pygame.display.update()

    def draw_background(self):
        self.main.screen.blit(self.background_image, (0, 0))

    def draw_sound_music_buttons(self):
        # draw sound button
        if self.main.sound_on:
            self.button_sound.image = self.sound_on_image
            self.button_sound.draw()
        else:
            self.button_sound.image = self.sound_off_image
            self.button_sound.draw()

        # draw music button
        if self.main.music_on:
            self.button_music.image = self.music_on_image
            self.button_music.draw()
        else:
            self.button_music.image = self.music_off_image
            self.button_music.draw()

    def check_sound_music_buttons_interactions(self):
        if self.button_sound.check_collision():
            self.mute_audio()
        if self.button_music.check_collision():
            self.mute_music()

    def mute_audio(self):
        if self.main.sound_on:
            self.main.sound_on = False
            self.main.last_global_volume = self.main.global_volume
            self.main.global_volume = 0
        else:
            self.main.sound_on = True
            self.main.global_volume = self.main.last_global_volume
            self.main.last_global_volume = 0

    def mute_music(self):
        if self.main.music_on:
            self.main.music_on = False
            self.main.last_music_volume = self.main.music_volume
            self.main.music_volume = 0
        else:
            self.main.music_on = True
            self.main.music_volume = self.main.last_music_volume
            self.main.last_music_volume = 0


class MainMenu:
    def __init__(self, menu):
        self.menu = menu

        # logo
        self.logo_image = pygame.image.load('assets/sprites/LogoGlow.png').convert_alpha()
        self.logo_image = pygame.transform.smoothscale(self.logo_image, (int(self.logo_image.get_size()[0] * 0.8), int(self.logo_image.get_size()[1] * 0.8)))

        # buttons images
        self.play_game_image = pygame.image.load('assets/sprites/ButtonPlayGame.png').convert_alpha()
        self.shop_image = pygame.image.load('assets/sprites/ButtonShop.png').convert_alpha()
        self.settings_image = pygame.image.load('assets/sprites/ButtonCredits.png').convert_alpha()
        self.quit_image = pygame.image.load('assets/sprites/ButtonQuit.png').convert_alpha()
        self.sound_on_image = pygame.image.load('assets/sprites/SoundOn.png').convert_alpha()

        # buttons
        self.button_play_game = tools.Button(self.menu.main.screen, self.play_game_image, (settings.WIDTH / 2, 425), 'center')
        self.button_shop = tools.Button(self.menu.main.screen, self.shop_image, (settings.WIDTH / 2, 484), 'center')
        self.button_settings = tools.Button(self.menu.main.screen, self.settings_image, (settings.WIDTH / 2, 536), 'center')
        self.button_quit = tools.Button(self.menu.main.screen, self.quit_image, (settings.WIDTH / 2, 589), 'center')
        self.button_sound_on = tools.Button(self.menu.main.screen, self.sound_on_image, (1234, 0))

    def update(self):
        self.check_events()

        self.menu.draw_background()
        self.draw_logo()
        self.draw_buttons()

        self.check_buttons_interactions()

    def check_events(self):
        for event in pygame.event.get():  # go through all events
            # quit
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # inputs key down
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.menu.main.playing = True
                    self.menu.main.menu = False  # TODO REMOVE THIS LATER / JUST FOR TESTING

    def draw_logo(self):
        self.menu.main.screen.blit(self.logo_image, (settings.WIDTH // 2 - self.logo_image.get_size()[0] // 2, 70))

    def draw_buttons(self):
        self.button_play_game.draw()
        self.button_shop.draw()
        self.button_settings.draw()
        self.button_quit.draw()
        self.menu.draw_sound_music_buttons()

    def check_buttons_interactions(self):
        if self.button_play_game.check_collision():
            self.menu.main.game = game.Game(self.menu.main)  # instantiate game
            self.menu.main.playing = True
            self.menu.main.in_menu = False
        if self.button_shop.check_collision():
            self.menu.current_menu = 'Shop'
        if self.button_settings.check_collision():
            self.menu.current_menu = 'Credits'
        if self.button_quit.check_collision():
            pygame.event.post(pygame.event.Event(pygame.QUIT))
        self.menu.check_sound_music_buttons_interactions()


class Credits:
    def __init__(self, menu):
        self.menu = menu

        self.text_y_offset = 230

        self.paused_screen_surface = pygame.Surface((settings.WIDTH, settings.HEIGHT), pygame.SRCALPHA, 32)

        # logo
        self.logo_image = pygame.image.load('assets/sprites/LogoGlow.png').convert_alpha()
        self.logo_image = pygame.transform.smoothscale(self.logo_image, (int(self.logo_image.get_size()[0] * 0.8), int(self.logo_image.get_size()[1] * 0.8)))

        # buttons images
        self.back_image = pygame.image.load('assets/sprites/ButtonBack.png').convert_alpha()

        # buttons
        self.button_back = tools.Button(self.menu.main.screen, self.back_image, (20, 20))

    def update(self):
        self.check_events()

        self.menu.draw_background()

        self.paused_screen_surface.fill((0, 0, 0, 100))
        self.menu.main.screen.blit(self.paused_screen_surface, (0, 0))

        self.draw_logo()
        self.draw_buttons()

        tools.draw_text(self.menu.main.screen, 'Programming, GUI and logo by', 'left', 32, (settings.WIDTH * 5.6 / 10, 0 + self.text_y_offset))
        tools.draw_text(self.menu.main.screen, 'Gustavo Pauli', 'left', 32, (settings.WIDTH * 5.6 / 10, 32 + self.text_y_offset))
        tools.draw_text(self.menu.main.screen, 'Art mainly done by', 'left', 32, (settings.WIDTH * 5.6 / 10, 90 + self.text_y_offset))
        tools.draw_text(self.menu.main.screen, 'DarkLava', 'left', 32, (settings.WIDTH * 5.6 / 10, 122 + self.text_y_offset))
        tools.draw_text(self.menu.main.screen, 'New Athletic M54 font by', 'left', 32, (settings.WIDTH * 5.6 / 10, 180 + self.text_y_offset))
        tools.draw_text(self.menu.main.screen, 'justme54s', 'left', 32, (settings.WIDTH * 5.6 / 10, 212 + self.text_y_offset))
        tools.draw_text(self.menu.main.screen, 'Special thanks to HalfBrick for', 'left', 32, (settings.WIDTH * 5.6 / 10, 270 + self.text_y_offset))
        tools.draw_text(self.menu.main.screen, 'Jetpack Joyride', 'left', 32, (settings.WIDTH * 5.6 / 10, 302 + self.text_y_offset))

        self.check_buttons_interactions()

    def check_events(self):
        for event in pygame.event.get():  # go through all events
            # quit
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # inputs key down
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.menu.current_menu = 'MainMenu'

    def draw_logo(self):
        self.menu.main.screen.blit(self.logo_image, (settings.WIDTH * 2.8 // 10 - self.logo_image.get_size()[0] // 2,
                                                     settings.HEIGHT // 2 - self.logo_image.get_size()[1] // 2))

    def draw_buttons(self):
        self.menu.draw_sound_music_buttons()
        self.button_back.draw()

    def check_buttons_interactions(self):
        self.menu.check_sound_music_buttons_interactions()
        if self.button_back.check_collision():
            self.menu.current_menu = 'MainMenu'


class Shop:
    def __init__(self, menu):
        self.menu = menu

        self.transparency_surface = pygame.Surface((settings.WIDTH, settings.HEIGHT), pygame.SRCALPHA, 32)

        # logo
        self.logo_image = pygame.image.load('assets/sprites/LogoGlow.png').convert_alpha()
        self.logo_image = pygame.transform.smoothscale(self.logo_image, (int(self.logo_image.get_size()[0] * 0.4), int(self.logo_image.get_size()[1] * 0.4)))

        # buttons images
        self.sound_on_image = pygame.image.load('assets/sprites/SoundOn.png').convert_alpha()
        self.back_image = pygame.image.load('assets/sprites/ButtonBack.png').convert_alpha()

        # buttons
        self.button_sound_on = tools.Button(self.menu.main.screen, self.sound_on_image, (1234, 0))
        self.button_back = tools.Button(self.menu.main.screen, self.back_image, (20, 20))

    def update(self):
        self.check_events()

        self.menu.draw_background()

        self.transparency_surface.fill((0, 0, 0, 100))
        self.menu.main.screen.blit(self.transparency_surface, (0, 0))

        self.draw_logo()
        self.draw_buttons()

        # tools.draw_text(self.menu.main.screen, 'Shop', 'center', 32, (settings.WIDTH / 2, 200))

        self.check_buttons_interactions()

    def check_events(self):
        for event in pygame.event.get():  # go through all events
            # quit
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # inputs key down
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.menu.current_menu = 'MainMenu'

    def draw_logo(self):
        self.menu.main.screen.blit(self.logo_image, (settings.WIDTH // 2 - self.logo_image.get_size()[0] // 2,
                                                     0))

    def draw_buttons(self):
        self.menu.draw_sound_music_buttons()
        self.button_back.draw()

    def check_buttons_interactions(self):
        self.menu.check_sound_music_buttons_interactions()
        if self.button_back.check_collision():
            self.menu.current_menu = 'MainMenu'
