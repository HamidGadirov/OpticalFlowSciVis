import taichi as ti
@ti.data_oriented
class Mod_AttrAllo:
    def add_attr(self, name, attr):
            if name in self.__dict__:
                print(
                    f"Warning: {name} already exists in {self.__class__.__name__}")
            else:
                if (isinstance(attr, list)):
                    self.__dict__[name] = []
                    self.m_attr_list[name] = []
                    for attr_i in attr:
                        self.__dict__[name].append(attr_i)

                else:
                    self.__dict__[name] = attr

                self.m_attr_list[name] = self.__dict__[name]

    # This function is deprecated (has been merged with add_attr()) and will be removed in the future.
    def add_attrs(self, name, attrs):
        if name in self.__dict__:
            print(
                f"Warning: {name} already exists in {self.__class__.__name__}")
        else:
            self.__dict__[name] = []
            self.m_attr_list[name] = []
            for attr in attrs:
                self.__dict__[name].append(attr)
            self.m_attr_list[name] = self.__dict__[name]

    def add_array(self, name, array, bundle=1):
        if name in self.__dict__:
            print(
                f"Warning: {name} already exists in {self.__class__.__name__}")
        else:
            if (isinstance(array, list)):
                self.__dict__[name] = []
                self.m_array_list[name] = []
                if (bundle == 1):
                    for array_i in array:
                        ti.root.dense(ti.i, self.m_part_num[None]).place(array_i)
                        self.__dict__[name].append(array_i)
                else:
                    for array_i in array:
                        ti.root.dense(ti.ij, (self.m_part_num[None], bundle)).place(array_i)
                        self.__dict__[name].append(array_i)
            else:
                if (bundle == 1):
                    ti.root.dense(ti.i, self.m_part_num[None]).place(array)
                else:
                    ti.root.dense(ti.ij, (self.m_part_num[None], bundle)).place(array)
                self.__dict__[name] = array

            self.m_array_list[name] = self.__dict__[name]

    # This function is deprecated (has been merged with add_array()) and will be removed in the future.
    def add_arrays(self, name, arrays):
        if name in self.__dict__:
            print(
                f"Warning: {name} already exists in {self.__class__.__name__}")
        else:
            self.__dict__[name] = []
            self.m_array_list[name] = []
            for array in arrays:
                ti.root.dense(ti.i, self.m_part_num[None]).place(array)
                self.__dict__[name].append(array)
            self.m_array_list[name] = self.__dict__[name]

    def add_struct(self, name, struct, bundle=1):
        if name in self.__dict__:
            print(
                f"Warning: {name} already exists in {self.__class__.__name__}")
        else:
            if (isinstance(struct, list)):
                self.__dict__[name] = []
                self.m_struct_list[name] = []
                if (bundle == 1):
                    for struct_i in struct:
                        self.__dict__[name].append(struct_i.field(shape=(self.m_part_num[None],)))
                else:
                    for struct_i in struct:
                        self.__dict__[name].append(struct_i.field(shape=(self.m_part_num[None], bundle)))
            else:
                if (bundle == 1):
                    self.__dict__[name] = struct.field(shape=(self.m_part_num[None],))
                else:
                    self.__dict__[name] = struct.field(shape=(self.m_part_num[None], bundle))
            self.m_struct_list[name] = self.__dict__[name]

    def add_structs(self, name, structs):
        if name in self.__dict__:
            print(
                f"Warning: {name} already exists in {self.__class__.__name__}")

        self.__dict__[name] = []
        self.m_struct_list[name] = []
        for struct in structs:
            self.__dict__[name].append(struct.field(shape=(self.m_part_num[None],)))
        self.m_struct_list[name] = self.__dict__[name]

    def instantiate_from_template(self, template, world):
        template(self, world)

    def verbose_structs(self, append=None):
        if append is not None:
            print("")
            text_color_begin = "\033[4;31;43m"
            text_color_end = "\033[0m"
            print(f"{text_color_begin}{append}{text_color_end}")
        sub_element_template = "  {0:16}\t|\t{1:8}\t|\t{2:16}"
        head_template = "{0:16}\t|"
        tilde_template = "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        dash_template = "---------------------------------------------------------------------------------------"
        print(tilde_template.format())
        print(f"{self.__class__.__name__} structs: ")
        print(tilde_template.format())
        for name in self.__dict__:
            if name in self.m_struct_list:
                if (isinstance(self.__dict__[name], list)):
                    iter = 0
                    for struct in self.__dict__[name]:
                        print(head_template.format(f"{name}[{iter}] {struct.shape}"))
                        for key in struct.keys:
                            # print('self.' + name + '[' + str(iter) + ']' + '.' + key  + '.dtype')
                            print(sub_element_template.format(
                                key,
                                f"dtype={eval('self.' + name + '[' + str(iter) + ']' + '.' + key  + '.dtype')}",
                                f"{[eval('self.' + name + '[' + str(iter) + ']' + '.' + key)]}"))
                        iter += 1
                        print(dash_template.format())
                else:
                    print(head_template.format(f"{name}{self.__dict__[name].shape}"))
                    for key in self.__dict__[name].keys:
                        print(sub_element_template.format(
                            key, f"dtype={eval('self.'+name+'.'+key).dtype}", f"{[eval('self.'+name+'.'+key)]}"))
                    print(dash_template.format())

    def verbose_arrays(self, append=None):
        if append is not None:
            print("")
            text_color_begin = "\033[4;31;43m"
            text_color_end = "\033[0m"
            print(f"{text_color_begin}{append}{text_color_end}")
        element_template = "{0:16}\t|\t{1:8}\t|\t{2:16}"
        sub_element_template = "  {0:16}\t|\t{1:8}\t|\t{2:16}"
        head_template = "{0:16}\t|"
        tilde_template = "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        dash_template = "---------------------------------------------------------------------------------------"
        print(tilde_template.format())
        print(f"{self.__class__.__name__} arrays:")
        print(tilde_template.format())
        for name in self.__dict__:
            if name in self.m_array_list:
                if (isinstance(self.__dict__[name], list)):
                    print(head_template.format(f"{name}"))
                    iter = 0
                    for array in self.__dict__[name]:
                        print(sub_element_template.format(
                            f"[{iter}] {array.shape}", f"dtype={array.dtype}", f"{[array]}"))
                        iter += 1
                else:
                    print(element_template.format(
                        f"{name} {self.__dict__[name].shape}", f"dtype={self.__dict__[name].dtype}", f"{[self.__dict__[name]]}"))
                print(dash_template.format())
        print(" ")

    def verbose_attrs(self, append=None):
        if append is not None:
            print("")
            text_color_begin = "\033[4;31;43m"
            text_color_end = "\033[0m"
            print(f"{text_color_begin}{append}{text_color_end}")
        element_template = "{0:16}\t|\t{1:8}\t|\t{2:16}"
        sub_element_template = "  {0:16}\t|\t{1:8}\t|\t{2:16}"
        head_template = "{0:16}\t|"
        tilde_template = "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        dash_template = "---------------------------------------------------------------------------------------"
        print(tilde_template.format())
        print(f"{self.__class__.__name__} attrs:")
        print(tilde_template.format())
        for name in self.__dict__:
            if name in self.m_attr_list:
                if (isinstance(self.__dict__[name], list)):
                    print(head_template.format(name))
                    iter = 0
                    for attr in self.__dict__[name]:
                        print(sub_element_template.format(
                            f"[{iter}]", f"type={type(attr)}", f"{attr}"))
                        iter += 1
                else:
                    print(element_template.format(
                        name, f"type={type(self.__dict__[name])}", f"{self.__dict__[name]}"))
                print(dash_template.format())
        print(" ")