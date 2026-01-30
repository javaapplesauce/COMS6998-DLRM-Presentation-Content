from manim import *

class DINOArchitecture(Scene):
    def construct(self):
        # --- 1. Styling Constants ---
        S_COLOR, S_FILL = PURPLE_A, PURPLE_E
        T_COLOR, T_FILL = GOLD_A, MAROON_E
        F_SIZE = 22
        
        # --- 2. Input Image Group ---
        # Fixed point 3: Small gap between image and crops
        img_rect = RoundedRectangle(height=1.2, width=1.2, color=WHITE, fill_opacity=0.1)
        img_label = Text("image", font_size=F_SIZE)
        input_source = VGroup(img_rect, img_label).to_edge(LEFT, buff=0.5)

        # --- 3. Functional Components Helper ---
        def create_pipeline(row_y, color, fill, enc_name, view_label):
            # View Box
            view = RoundedRectangle(height=1.4, width=1.8, color=color, fill_opacity=0.3, fill_color=fill)
            v_txt = Text(view_label, font_size=F_SIZE-6).move_to(view)
            view_grp = VGroup(view, v_txt).move_to([ -3, row_y, 0])

            # CLS Token (Point 4: Image goes straight to CLS)
            cls_bar = Rectangle(height=1.4, width=0.2, color=color, fill_opacity=0.5, fill_color=fill)
            cls_top = Square(side_length=0.3, color=color, fill_opacity=0.8, fill_color=fill).next_to(cls_bar, UP, buff=0.05)
            cls_lbl = Text("CLS", font_size=F_SIZE-8, color=color).next_to(cls_top, UP, buff=0.05)
            cls_grp = VGroup(cls_bar, cls_top, cls_lbl).next_to(view_grp, RIGHT, buff=0.6)

            # Encoder (Point 1: Text inside/under correctly)
            enc = RoundedRectangle(height=1.4, width=2.0, color=color, fill_opacity=0.3, fill_color=fill).next_to(cls_grp, RIGHT, buff=0.6)
            e_txt = Text(f"Encoder\n({enc_name})", font_size=F_SIZE-4).move_to(enc)
            enc_grp = VGroup(enc, e_txt)

            # Internal Straight Green Arrows (Point 5)
            a1 = Line(view.get_right(), cls_grp.get_left(), color=GREEN_B).add_tip(tip_length=0.1)
            a2 = Line(cls_grp.get_right(), enc.get_left(), color=GREEN_B).add_tip(tip_length=0.1)
            
            return VGroup(view_grp, cls_grp, enc_grp, a1, a2)

        student_row = create_pipeline(2.2, S_COLOR, S_FILL, "Student", "Cropped View")
        teacher_row = create_pipeline(-2.2, T_COLOR, T_FILL, "Teacher", "Global View")

        # --- 4. EMA Update (Point 2: Thinner arrow, Bold text) ---
        ema_arrow = Line(student_row[2].get_bottom(), teacher_row[2].get_top(), color=WHITE, stroke_width=1.5).add_tip(tip_length=0.1)
        ema_label = Text("EMA", font_size=F_SIZE, weight=BOLD, color=GREEN_B).next_to(ema_arrow, RIGHT, buff=0.2)

        # --- 5. Centering & Softmax (Point 7: Horizontal bar plots) ---
        centering = RoundedRectangle(height=0.7, width=1.6, color=T_COLOR, fill_opacity=0.2).next_to(teacher_row[2], RIGHT, buff=0.6)
        c_txt = Text("Centering", font_size=F_SIZE-8).move_to(centering)

        def create_softmax_stack(target, color, temp_val):
            # Softmax Box
            soft = RoundedRectangle(height=0.5, width=1.1, color=color, fill_opacity=0.2).next_to(target, RIGHT, buff=0.7)
            s_txt = Text("Softmax", font_size=F_SIZE-10).move_to(soft)
            t_txt = Text(f"T = {temp_val}", font_size=F_SIZE-8).next_to(soft, UP, buff=0.1)
            
            # Bar Chart (Corrected baseline, aligned horizontally)
            bar_heights = [0.1, 0.5, 0.3, 0.05, 0.2]
            bars = VGroup()
            for h in bar_heights:
                b = Rectangle(height=h, width=0.1, color=ORANGE, fill_opacity=0.8, stroke_width=1)
                b.align_to(soft, DOWN) # Aligns all bars to a flat baseline
                bars.add(b)
            bars.arrange(RIGHT, buff=0.05, aligned_edge=DOWN).next_to(soft, RIGHT, buff=0.4)
            
            # Connection Arrow (Point 6)
            arrow = Line(target.get_right(), soft.get_left(), color=WHITE).add_tip(tip_length=0.1)
            return VGroup(soft, s_txt, t_txt, bars, arrow)

        soft_s = create_softmax_stack(student_row[2], S_COLOR, "0.1")
        soft_t = create_softmax_stack(centering, T_COLOR, "0.04")
        stop_grad = Text("// s.g.", font_size=F_SIZE-10, color=RED).next_to(soft_t[0], RIGHT, buff=0.05)

        # --- 6. Loss Mechanism ---
        loss_box = RoundedRectangle(height=0.9, width=1.4, color=WHITE, fill_opacity=0.1)
        loss_txt = Text("Loss", font_size=F_SIZE).move_to(loss_box)
        loss_grp = VGroup(loss_box, loss_txt).move_to([5.8, 0, 0])

        # Loss Comparison Arrows
        comp_s = Line(soft_s[3].get_bottom(), loss_box.get_top(), color=S_COLOR).add_tip(tip_length=0.1)
        comp_t = Line(soft_t[3].get_top(), loss_box.get_bottom(), color=T_COLOR).add_tip(tip_length=0.1)

        # --- 7. Final Assembly & Animation ---
        # Bracket shift to avoid "image" box overlap
        bracket = Brace(VGroup(student_row[0], teacher_row[0]), LEFT, buff=0.15).next_to(input_source, RIGHT, buff=0.1)
        
        # Scaling the whole scene to fit
        all_elements = VGroup(input_source, bracket, student_row, teacher_row, ema_arrow, ema_label, 
                              centering, c_txt, soft_s, soft_t, stop_grad, loss_grp, comp_s, comp_t)
        all_elements.scale(0.7).center()

        # Render Sequence
        self.play(FadeIn(input_source), Create(bracket))
        self.play(Create(student_row), Create(teacher_row), run_time=2.5)
        self.play(Write(ema_label), Create(ema_arrow))
        self.play(FadeIn(centering, c_txt))
        self.play(Create(soft_s[4]), Create(soft_t[4])) # Arrows to softmax
        self.play(FadeIn(soft_s[:4]), FadeIn(soft_t[:4]), Write(stop_grad))
        self.play(Create(loss_grp), Create(comp_s), Create(comp_t))
        self.wait(3)