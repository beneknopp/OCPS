class RequestParamsParser:

    @classmethod
    def parse_activity_selected_types(cls, form):
        activity_selected_types = {}
        act_keys = [key for key in form if key[:4] == "act:"]
        for act_key in act_keys:
            act = act_key[4:]
            types_string = form[act_key]
            types = types_string.split(",")
            activity_selected_types[act] = types
        return activity_selected_types

    @classmethod
    def parse_activity_leading_type_and_selected_types(cls, form):
        activity_leading_types = {}
        activity_selected_types = {}
        act_keys = [key for key in form if key[:4] == "act:"]
        for act_key in act_keys:
            act = act_key[4:]
            types_string = form[act_key]
            types = types_string.split(",")
            leading_type = types[0]
            activity_leading_types[act] = leading_type
            activity_selected_types[act] = types
        return activity_leading_types, activity_selected_types
